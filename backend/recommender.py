"""Recommendation engine: vibe-vector cosine matching + optional live enrichment."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd

from backend.vibe_taxonomy import (
    VIBES,
    clean_tags,
    primary_vibe,
    tags_to_vibe_vector,
    top_vibes,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

TargetType = Literal["all", "book", "song"]
InputKind = Literal["book", "song", "auto"]


@dataclass
class RecItem:
    title: str
    creator: str
    type: str
    similarity: float
    primary_vibe: str
    vibe_labels: str
    why: str = ""
    source: str = "catalog"


def _pretty_creator(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "Unknown"
    if " " in s or "," in s:
        return s.title() if s.islower() else s
    # glued lowercase names: stephenking -> Stephen King (best-effort)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    if spaced != s:
        return spaced
    # common pattern first+last with no separator
    return s


class VibeRecommender:
    def __init__(self, data_dir: str = DATA):
        self.data_dir = data_dir
        self.df: pd.DataFrame
        self.matrix: np.ndarray
        self.index: dict[str, list[int]]
        self._load()

    def _load(self) -> None:
        parquet = os.path.join(self.data_dir, "catalog.parquet")
        pkl = os.path.join(self.data_dir, "catalog.pkl")
        if os.path.exists(parquet):
            self.df = pd.read_parquet(parquet)
        elif os.path.exists(pkl):
            self.df = pd.read_pickle(pkl)
        else:
            raise FileNotFoundError(
                "No catalog found. Run: python -m backend.build_catalog"
            )
        self.matrix = np.load(os.path.join(self.data_dir, "vibe_matrix.npy"))
        with open(os.path.join(self.data_dir, "search_index.json"), encoding="utf-8") as f:
            self.index = json.load(f)

    @staticmethod
    def _is_companion_title(title: str) -> bool:
        t = title.lower()
        noise = (
            "guide",
            "cookbook",
            "coloring",
            "colouring",
            "journal",
            "trivia",
            "quiz",
            "box set",
            "boxed set",
            "ultimate",
            "companion",
            "encyclopedia",
            "dictionary",
            "workbook",
            "official",
            "behind the scenes",
        )
        return any(n in t for n in noise)

    def resolve_title(
        self,
        query: str,
        prefer_type: str | None = None,
    ) -> tuple[int, str] | None:
        """
        Resolve a user query to a catalog row.

        Important: never treat short catalog titles as substrings of the query
        (that made "UR" match "foURth wing").

        Franchise/series queries like "harry potter" match many titles — return
        None so live lookup can represent the series properly instead of
        randomly latching onto Goblet of Fire / a guidebook.
        """
        q = query.strip().lower()
        if not q:
            return None

        # 1) Exact title
        if q in self.index:
            idx = self._pick_index(self.index[q], prefer_type)
            return idx, str(self.df.iloc[idx]["title"])

        # 2) Query contained in a catalog title (safe direction)
        contains_hits: list[tuple[int, str, int]] = []
        for title_key, idxs in self.index.items():
            if q in title_key:
                idx = self._pick_index(idxs, prefer_type)
                if prefer_type and self.df.iloc[idx]["type"] != prefer_type:
                    # still allow, but we'll prefer matching type via _pick_index
                    pass
                title = str(self.df.iloc[idx]["title"])
                if prefer_type and str(self.df.iloc[idx]["type"]) != prefer_type:
                    continue
                contains_hits.append((idx, title, len(title_key)))

        if contains_hits:
            main_hits = [h for h in contains_hits if not self._is_companion_title(h[1])]
            pool = main_hits or []

            # Short franchise-style queries ("harry potter", "percy jackson") usually
            # match many longer titles. Prefer live series lookup over a random volume.
            short_franchise = len(q.split()) <= 3
            near_exact = [h for h in pool if h[2] <= len(q) + 10]
            if short_franchise and not near_exact and (len(contains_hits) >= 2 or not pool):
                return None
            if len(pool) >= 3 and short_franchise:
                return None
            if not pool:
                return None

            # Prefer book 1 / shorter core titles over mid-series volumes
            def rank(hit: tuple[int, str, int]) -> tuple:
                title = hit[1].lower()
                series_num = 99
                m = re.search(r"#\s*(\d+)", title)
                if m:
                    series_num = int(m.group(1))
                for marker, n in (
                    ("philosopher", 1),
                    ("sorcerer", 1),
                    ("lightning thief", 1),
                    ("the sea of monsters", 2),
                ):
                    if marker in title:
                        series_num = min(series_num, n)
                starts = 0 if title.startswith(q) else 1
                return (series_num, starts, hit[2], abs(hit[2] - len(q)))

            pool.sort(key=rank)
            idx, title, _ = pool[0]
            return idx, title

        # 3) Strict word overlap — ignore stopwords so
        # "chasing that feeling" cannot latch onto "...Rainy Day Feeling Again"
        stop = {
            "that",
            "this",
            "with",
            "from",
            "your",
            "have",
            "been",
            "were",
            "they",
            "them",
            "then",
            "than",
            "when",
            "what",
            "come",
            "comes",
            "again",
            "into",
            "just",
            "like",
            "love",
            "baby",
            "girl",
            "time",
            "life",
            "song",
            "night",
            "heart",
            "feeling",  # too common in song titles
            "forever",
            "young",
            "little",
            "about",
        }
        q_words = {w for w in re.findall(r"[a-z0-9]{4,}", q)}
        q_content = {w for w in q_words if w not in stop and len(w) >= 5}
        if not q_content:
            # No distinctive tokens → don't fuzzy-match catalog; use live lookup
            return None

        best: tuple[int, str, float] | None = None
        for title_key, idxs in self.index.items():
            t_words = set(re.findall(r"[a-z0-9]{4,}", title_key))
            if not t_words:
                continue
            # Every distinctive query word must appear in the catalog title
            if not q_content.issubset(t_words):
                continue
            overlap = len(q_words & t_words) / max(len(q_words), 1)
            if overlap < 0.8:
                continue
            idx = self._pick_index(idxs, prefer_type)
            title = str(self.df.iloc[idx]["title"])
            if self._is_companion_title(title):
                continue
            if prefer_type and self.df.iloc[idx]["type"] != prefer_type:
                continue
            score = overlap + 0.15 * len(q_content & t_words)
            if best is None or score > best[2]:
                best = (idx, title, score)
        if best:
            return best[0], best[1]
        return None

    def _pick_index(self, idxs: list[int], prefer_type: str | None) -> int:
        if not prefer_type:
            return idxs[0]
        for i in idxs:
            if self.df.iloc[i]["type"] == prefer_type:
                return i
        return idxs[0]

    def search_suggestions(
        self,
        query: str,
        limit: int = 12,
        type_filter: str | None = None,
    ) -> list[dict]:
        q = query.strip().lower()
        if len(q) < 2:
            return []
        out = []
        for _, row in self.df.iterrows():
            title = str(row["title"])
            creator = _pretty_creator(str(row.get("creator_display") or row["creator"]))
            title_l = title.lower()
            # Only suggest when query appears in the title/creator (not reverse)
            if q not in title_l and q not in creator.lower():
                continue
            if type_filter and row["type"] != type_filter:
                continue
            out.append(
                {
                    "title": title,
                    "creator": creator,
                    "type": row["type"],
                    "primary_vibe": row.get("primary_vibe", ""),
                }
            )
            if len(out) >= limit:
                break
        return out

    def _why(self, vibe_labels: str, primary: str) -> str:
        labels = [x.strip() for x in (vibe_labels or "").split(",") if x.strip()]
        if not labels:
            return f"Shares a similar {primary or 'overall'} mood."
        shown = ", ".join(labels[:3])
        return f"Similar mood: {shown}."

    def recommend_from_vector(
        self,
        vector: list[float] | np.ndarray,
        target_type: TargetType = "all",
        top_n: int = 6,
        exclude_idx: int | None = None,
        min_score: float = 0.28,
    ) -> list[RecItem]:
        v = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            return []
        v = v / n
        scores = self.matrix @ v

        order = np.argsort(-scores)
        results: list[RecItem] = []
        for i in order:
            i = int(i)
            if exclude_idx is not None and i == exclude_idx:
                continue
            item_type = self.df.iloc[i]["type"]
            if target_type != "all" and item_type != target_type:
                continue
            score = float(scores[i])
            if score < min_score:
                break
            row = self.df.iloc[i]
            labels = str(row.get("vibe_labels", ""))
            primary = str(row.get("primary_vibe", ""))
            results.append(
                RecItem(
                    title=str(row["title"]),
                    creator=_pretty_creator(str(row.get("creator_display") or row["creator"])),
                    type=str(row["type"]),
                    similarity=round(score, 4),
                    primary_vibe=primary,
                    vibe_labels=labels,
                    why=self._why(labels, primary),
                    source="catalog",
                )
            )
            if len(results) >= top_n:
                break
        return results

    def recommend(
        self,
        title: str,
        target_type: TargetType = "all",
        top_n: int = 6,
        prefer_type: str | None = None,
    ) -> dict[str, Any]:
        resolved = self.resolve_title(title, prefer_type=prefer_type)
        if not resolved:
            return {"found": False, "query": title, "recommendations": []}

        idx, canonical = resolved
        row = self.df.iloc[idx]
        query_vec = self.matrix[idx]
        q_primary = str(row.get("primary_vibe", ""))
        q_labels = str(row.get("vibe_labels", ""))

        if target_type == "all":
            opposite = "song" if row["type"] == "book" else "book"
            cross = self.recommend_from_vector(
                query_vec,
                target_type=opposite,
                top_n=max(4, (top_n + 1) // 2),
                exclude_idx=idx,
            )
            same = self.recommend_from_vector(
                query_vec,
                target_type=row["type"],
                top_n=max(3, top_n // 2),
                exclude_idx=idx,
            )
            recs = cross + same
        else:
            recs = self.recommend_from_vector(
                query_vec, target_type=target_type, top_n=top_n, exclude_idx=idx
            )

        return {
            "found": True,
            "query": title,
            "matched_title": canonical,
            "matched_creator": _pretty_creator(str(row.get("creator_display") or row["creator"])),
            "matched_type": str(row["type"]),
            "primary_vibe": q_primary,
            "vibe_labels": q_labels,
            "description": "",
            "source": "catalog",
            "recommendations": [r.__dict__ for r in recs],
        }

    def recommend_from_live_item(
        self,
        *,
        title: str,
        creator: str,
        item_type: str,
        tags: list[str],
        description: str = "",
        target_type: TargetType = "all",
        top_n: int = 6,
    ) -> dict[str, Any]:
        cleaned = clean_tags(tags)
        text = f"{title} {creator} {description} {' '.join(cleaned)}"
        vec = tags_to_vibe_vector(cleaned, text=text)
        q_primary = primary_vibe(vec)
        q_labels = ", ".join(top_vibes(vec, 4))

        if target_type == "all":
            opposite = "song" if item_type == "book" else "book"
            recs = self.recommend_from_vector(
                vec, target_type=opposite, top_n=max(4, (top_n + 1) // 2)
            )
            recs += self.recommend_from_vector(
                vec, target_type=item_type, top_n=max(3, top_n // 2)
            )
        else:
            recs = self.recommend_from_vector(vec, target_type=target_type, top_n=top_n)

        return {
            "found": True,
            "query": title,
            "matched_title": title,
            "matched_creator": creator,
            "matched_type": item_type,
            "primary_vibe": q_primary,
            "vibe_labels": q_labels,
            "description": (description or "").strip(),
            "source": "live",
            "live_tags": cleaned[:20],
            "recommendations": [r.__dict__ for r in recs],
        }


@lru_cache(maxsize=1)
def get_recommender() -> VibeRecommender:
    return VibeRecommender()


def clear_recommender_cache() -> None:
    get_recommender.cache_clear()
