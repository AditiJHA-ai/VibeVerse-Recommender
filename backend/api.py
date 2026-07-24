"""FastAPI backend for VibeVerse."""

from __future__ import annotations

import os
import re
from typing import Literal, Optional

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.enrichment import enrich
from backend.recommender import get_recommender
from backend.vibe_taxonomy import VIBES

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

app = FastAPI(title="VibeVerse API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    # What the user typed: a book title or a song title
    input_type: Literal["book", "song"] = "book"
    # What they want back
    want: Literal["all", "book", "song"] = "all"
    top_n: int = Field(8, ge=1, le=20)
    allow_live: bool = True

    # backwards-compatible aliases
    target_type: Optional[Literal["all", "book", "song"]] = None
    prefer: Optional[Literal["book", "song"]] = None


def _has_real_key(name: str) -> bool:
    v = (os.getenv(name) or "").strip().strip('"').strip("'")
    if len(v) < 8:
        return False
    low = v.lower()
    return not (
        low.startswith("your_")
        or low.startswith("changeme")
        or low in {"example", "xxx", "todo", "replace_me"}
    )


@app.get("/api/health")
def health():
    rec = get_recommender()
    return {
        "ok": True,
        "items": len(rec.df),
        "vibes": VIBES,
        "google_books": _has_real_key("GOOGLE_BOOKS_API_KEY"),
        "lastfm": _has_real_key("LASTFM_API_KEY"),
    }


@app.get("/api/vibes")
def list_vibes():
    return {"vibes": VIBES}


@app.get("/api/suggest")
def suggest(
    q: str = Query(..., min_length=2),
    type: Optional[Literal["book", "song"]] = None,
    limit: int = Query(12, ge=1, le=30),
):
    rec = get_recommender()
    return {"results": rec.search_suggestions(q, limit=limit, type_filter=type)}


def _query_matches_title(query: str, matched_title: str) -> bool:
    """True only when the matched catalog title is actually what the user asked for."""
    q = (query or "").strip().lower()
    t = (matched_title or "").strip().lower()
    if not q or not t:
        return False
    if q == t or q in t:
        return True
    # Only allow matched-title-inside-query when the match is long enough
    # (prevents "UR" matching inside "foURth wing")
    if len(t) >= 5 and t in q:
        return True
    # series wrapper: "harry potter" vs "Harry Potter series"
    if t.endswith(" series") and (q == t[: -len(" series")] or q in t):
        return True
    stop = {
        "that",
        "this",
        "with",
        "from",
        "your",
        "have",
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
        "feeling",
        "forever",
        "young",
        "little",
        "about",
        "the",
        "and",
        "for",
    }
    qw = {w for w in re.findall(r"[a-z0-9]{4,}", q) if w not in stop and len(w) >= 5}
    tw = set(re.findall(r"[a-z0-9]{4,}", t))
    if not qw:
        # short/generic query — require strong containment already handled above
        return False
    return qw.issubset(tw)


async def _attach_live_discovery(result: dict, want: str, top_n: int) -> dict:
    """Blend modern Last.fm / Google Books picks into recommendations."""
    from backend.discovery import (
        discover_books_by_vibes,
        discover_songs_by_vibes,
        vector_from_labels,
    )

    labels = result.get("vibe_labels") or result.get("primary_vibe") or ""
    tags = result.get("live_tags") or []
    text = f"{result.get('matched_title','')} {result.get('matched_creator','')}"
    qvec = vector_from_labels(labels, tags=tags, text=text)

    live_recs: list[dict] = []
    need_songs = want in ("all", "song")
    need_books = want in ("all", "book")

    if need_songs:
        live_recs.extend(
            await discover_songs_by_vibes(labels, qvec, top_n=top_n, per_tag=12)
        )
    if need_books:
        live_recs.extend(await discover_books_by_vibes(labels, qvec, top_n=max(4, top_n // 2)))

    # Prefer live (modern) results, then fill with catalog — dedupe by title+creator
    catalog = result.get("recommendations") or []
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in live_recs + catalog:
        key = (str(item.get("title", "")).lower(), str(item.get("creator", "")).lower())
        if not key[0] or key in seen:
            continue
        seen.add(key)
        merged.append(item)

    # Keep type balance when want=all
    if want == "all":
        songs = [x for x in merged if x.get("type") == "song"][:top_n]
        books = [x for x in merged if x.get("type") == "book"][: max(2, top_n // 2)]
        merged = songs + books
    else:
        merged = [x for x in merged if x.get("type") == want][:top_n]

    result["recommendations"] = merged
    # Keep internals for debugging only — never surface source labels to the UI copy
    result.pop("match_explanation", None)
    return result


@app.post("/api/recommend")
async def recommend(body: RecommendRequest):
    rec = get_recommender()

    input_type = body.prefer or body.input_type
    want = body.target_type or body.want

    result = rec.recommend(
        body.query,
        target_type=want,
        top_n=body.top_n,
        prefer_type=input_type,
    )

    # If catalog hit is the wrong medium (user said book, we found a song), use live instead
    if result.get("found") and result.get("matched_type") != input_type:
        result = {"found": False, "query": body.query, "recommendations": []}

    # Reject weak catalog title collisions (shared words ≠ same song/book)
    if result.get("found") and not _query_matches_title(body.query, result.get("matched_title") or ""):
        result = {"found": False, "query": body.query, "recommendations": []}

    if not result.get("found"):
        if not body.allow_live:
            raise HTTPException(status_code=404, detail="Title not in catalog")

        live = await enrich(body.query, prefer=input_type)
        if not live:
            raise HTTPException(
                status_code=404,
                detail="Couldn't find that title. Try the full name and try again.",
            )

        result = rec.recommend_from_live_item(
            title=live["title"],
            creator=live["creator"],
            item_type=live["type"],
            tags=live.get("tags") or [],
            description=live.get("description") or "",
            target_type=want,
            top_n=body.top_n,
        )
        result["description"] = (live.get("description") or "").strip()
        result["thumbnail"] = live.get("thumbnail")
        result["info_link"] = live.get("info_link") or live.get("url")

    # If catalog hit has no blurb, fill metadata using the SAME title + artist/author
    if result.get("found") and not result.get("description") and body.allow_live:
        try:
            from backend.enrichment import lookup_book, lookup_song

            title = result.get("matched_title") or body.query
            creator = result.get("matched_creator") or ""
            meta = None
            if (result.get("matched_type") or input_type) == "song":
                meta = await lookup_song(title, artist=creator or None)
            else:
                meta = await lookup_book(title)
            # Only accept blurbs that belong to this creator when it's a song
            if meta and meta.get("description"):
                meta_artist = (meta.get("creator") or "").lower()
                want_artist = creator.lower()
                ok = True
                if result.get("matched_type") == "song" and want_artist and meta_artist:
                    a, b = re.sub(r"[^a-z0-9]+", "", want_artist), re.sub(
                        r"[^a-z0-9]+", "", meta_artist
                    )
                    ok = a == b or a in b or b in a
                if ok:
                    result["description"] = meta["description"].strip()
                    result["thumbnail"] = result.get("thumbnail") or meta.get("thumbnail")
        except Exception:
            pass

    if body.allow_live:
        result = await _attach_live_discovery(result, want=want, top_n=max(body.top_n, 10))

    # Fold series mates / same-artist / related-artist tracks into main vibe matches
    if body.allow_live and result.get("found"):
        from backend.related import merge_affinity_into_recommendations

        result = await merge_affinity_into_recommendations(
            result, want=want, top_n=max(body.top_n, 10)
        )

    result.pop("match_explanation", None)
    result.pop("discovery", None)
    result.pop("related", None)
    return result


def main():
    import uvicorn

    uvicorn.run(
        "backend.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )


if __name__ == "__main__":
    main()
