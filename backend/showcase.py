"""Curated landing-page pairings that must always surface in search.

The marketing "Try These Pairings" section makes specific cross-domain claims.
Pure vibe cosine can't guarantee those, so we pin them when the query matches.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def _norm(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[“”\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _pin(
    *,
    title: str,
    creator: str,
    item_type: str,
    primary_vibe: str,
    why: str,
    similarity: float = 0.96,
) -> dict[str, Any]:
    return {
        "title": title,
        "creator": creator,
        "type": item_type,
        "similarity": similarity,
        "primary_vibe": primary_vibe,
        "vibe_labels": primary_vibe,
        "why": why,
        "source": "showcase",
    }


SHOWCASES: list[dict[str, Any]] = [
    {
        "aliases": (
            "the great gatsby",
            "great gatsby",
            "gatsby",
            "the great gatsby by f. scott fitzgerald",
            "the great gatsby by f scott fitzgerald",
        ),
        "input_type": "book",
        "seed": {
            "matched_title": "The Great Gatsby",
            "matched_creator": "F. Scott Fitzgerald",
            "matched_type": "book",
            "primary_vibe": "intimate",
            "vibe_labels": "intimate, melancholy, nostalgic",
            "tags": [
                "classics",
                "literary-fiction",
                "nostalgic",
                "melancholy",
                "romance",
                "jazz",
                "intimate",
            ],
            "description": (
                "Jazz-age glamour with melancholy undertones - a portrait of longing, "
                "excess, and bittersweet nostalgia."
            ),
        },
        "pins": [
            _pin(
                title="Sad Girl",
                creator="Lana Del Rey",
                item_type="song",
                primary_vibe="intimate",
                why="Atmospheric, nostalgic pop that mirrors Gatsby's bittersweet longing.",
                similarity=0.97,
            ),
            _pin(
                title="National Anthem",
                creator="Lana Del Rey",
                item_type="song",
                primary_vibe="intimate",
                why="Jazz-age glamour and melancholy - the same dreamy ache as Fitzgerald.",
                similarity=0.95,
            ),
            _pin(
                title="Ultraviolence",
                creator="Lana Del Rey",
                item_type="song",
                primary_vibe="epic",
                why="Lush, cinematic melancholy that matches the novel's haunted romance.",
                similarity=0.93,
            ),
            _pin(
                title="The Sun Also Rises",
                creator="Ernest Hemingway",
                item_type="book",
                primary_vibe="epic",
                why="Lost-generation glamour and melancholy - a natural literary neighbor to Gatsby.",
                similarity=0.92,
            ),
            _pin(
                title="The Curious Case of Benjamin Button",
                creator="F. Scott Fitzgerald",
                item_type="book",
                primary_vibe="dreamy",
                why="Same Fitzgerald voice - dreamy, bittersweet, haunted by time and longing.",
                similarity=0.9,
            ),
        ],
    },
    {
        "aliases": (
            "midnight by taylor swift",
            "midnights by taylor swift",
            "midnights",
            "midnights taylor swift",
            "midnight taylor swift",
            "taylor swift midnights",
            "taylor swift midnight",
        ),
        "input_type": "song",
        "seed": {
            "matched_title": "Midnights",
            "matched_creator": "Taylor Swift",
            "matched_type": "song",
            "primary_vibe": "intimate",
            "vibe_labels": "intimate, contemplative, melancholy",
            "tags": [
                "intimate",
                "melancholy",
                "contemplative",
                "indie-pop",
                "singer-songwriter",
                "pop",
            ],
            "description": (
                "Introspective late-night reflections layered with pop production - "
                "quiet confessions after dark."
            ),
        },
        "pins": [
            _pin(
                title="Normal People",
                creator="Sally Rooney",
                item_type="book",
                primary_vibe="intimate",
                why="Poetic intimacy and emotional depth - the literary twin of Midnights.",
                similarity=0.97,
            ),
            _pin(
                title="august",
                creator="Taylor Swift",
                item_type="song",
                primary_vibe="intimate",
                why="The same late-night, confessional pop atmosphere as Midnights.",
                similarity=0.94,
            ),
            _pin(
                title="the lakes - original version",
                creator="Taylor Swift",
                item_type="song",
                primary_vibe="intimate",
                why="Quiet, literary intimacy that sits beside Midnights' reflective mood.",
                similarity=0.92,
            ),
        ],
    },
    {
        "aliases": (
            "dune",
            "dune by frank herbert",
            "dune frank herbert",
            "frank herbert dune",
        ),
        "input_type": "book",
        "seed": {
            "matched_title": "Dune",
            "matched_creator": "Frank Herbert",
            "matched_type": "book",
            "primary_vibe": "epic",
            "vibe_labels": "epic, adventure, dreamy",
            "tags": [
                "science-fiction",
                "epic",
                "adventure",
                "fantasy",
                "orchestral",
                "dreamy",
            ],
            "description": (
                "Epic, expansive world-building with orchestral grandeur - desert empires, "
                "destiny, and awe at planetary scale."
            ),
        },
        # Do not treat Dune Messiah as this showcase seed.
        "reject_matched_titles": ("dune messiah",),
        "pins": [
            _pin(
                title="Time",
                creator="Hans Zimmer",
                item_type="song",
                primary_vibe="dreamy",
                why="Sweeping, atmospheric score energy that echoes Dune's sense of wonder.",
                similarity=0.97,
            ),
            _pin(
                title="Cornfield Chase",
                creator="Hans Zimmer",
                item_type="song",
                primary_vibe="intimate",
                why="Expansive orchestral wonder with the same epic emotional scale.",
                similarity=0.94,
            ),
            _pin(
                title="Dune Messiah (Dune Chronicles #2)",
                creator="Frank Herbert",
                item_type="book",
                primary_vibe="epic",
                why="The direct sequel - same epic desert mythology and political scale.",
                similarity=0.91,
            ),
        ],
    },
]


def match_showcase(query: str, input_type: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return the showcase entry for a query, if any."""
    q = _norm(query)
    if not q:
        return None

    # Strip trailing "by author" noise already covered by aliases, but also
    # allow close contains for slightly longer typed queries.
    for entry in SHOWCASES:
        aliases = entry["aliases"]
        if q in aliases:
            if input_type and input_type != entry["input_type"]:
                # Still allow if the alias is unambiguous (e.g. "midnights").
                if q not in aliases[:1] and len(q) < 12:
                    continue
            return entry
        for alias in aliases:
            if len(alias) >= 5 and (alias in q or q in alias):
                if input_type and input_type != entry["input_type"] and q != alias:
                    continue
                return entry
    return None


def apply_showcase(
    result: dict[str, Any],
    *,
    query: str,
    input_type: str,
    want: str = "all",
    top_n: int = 8,
) -> dict[str, Any]:
    """Force curated seed + pin claimed matches to the top of recommendations."""
    entry = match_showcase(query, input_type)
    if not entry:
        return result

    seed = entry["seed"]
    reject = { _norm(t) for t in entry.get("reject_matched_titles") or () }
    matched = _norm(str(result.get("matched_title") or ""))

    # Wrong catalog collision (e.g. Dune → Dune Messiah): replace seed metadata.
    if not result.get("found") or matched in reject or matched != _norm(seed["matched_title"]):
        result = {
            **result,
            "found": True,
            "query": query,
            "matched_title": seed["matched_title"],
            "matched_creator": seed["matched_creator"],
            "matched_type": seed["matched_type"],
            "primary_vibe": seed["primary_vibe"],
            "vibe_labels": seed["vibe_labels"],
            "description": seed.get("description") or result.get("description") or "",
            "source": "showcase",
            "recommendations": list(result.get("recommendations") or []),
        }
    else:
        # Keep catalog/live match, but ensure blurb exists.
        if not result.get("description") and seed.get("description"):
            result["description"] = seed["description"]
        result["matched_title"] = seed["matched_title"]
        result["matched_creator"] = seed["matched_creator"]
        result["matched_type"] = seed["matched_type"]

    pins = list(entry["pins"])
    if want in {"book", "song"}:
        pins = [p for p in pins if p["type"] == want]

    existing = list(result.get("recommendations") or [])
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []

    for item in pins + existing:
        key = (
            _norm(str(item.get("title") or "")),
            _norm(str(item.get("creator") or "")),
        )
        if not key[0] or key in seen:
            continue
        # Drop opposite-of-want filler when filtered.
        if want in {"book", "song"} and item.get("type") != want:
            continue
        seen.add(key)
        merged.append(item)

    if want == "all":
        songs = [x for x in merged if x.get("type") == "song"]
        books = [x for x in merged if x.get("type") == "book"]
        n_songs = max(4, (top_n + 1) // 2)
        n_books = max(3, top_n // 2)
        result["recommendations"] = songs[:n_songs] + books[:n_books]
    else:
        result["recommendations"] = merged[: max(top_n, len(pins))]
    result["showcase"] = True
    return result
