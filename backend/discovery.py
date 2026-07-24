"""Live discovery — pull modern songs/books from Last.fm + Google Books by vibe."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import httpx
import numpy as np

from backend.vibe_taxonomy import (
    VIBE_INDEX,
    clean_tags,
    primary_vibe,
    tags_to_vibe_vector,
    top_vibes,
)

LASTFM = "https://ws.audioscrobbler.com/2.0/"
GOOGLE_BOOKS = "https://www.googleapis.com/books/v1/volumes"

# Map our vibes → Last.fm tags that actually surface current music
VIBE_TO_LASTFM_TAGS: dict[str, list[str]] = {
    "intimate": ["singer-songwriter", "indie", "acoustic", "soft rock"],
    "electric": ["pop", "dance", "electropop", "indie pop"],
    "dreamy": ["dream pop", "indie", "atmospheric", "shoegaze"],
    "epic": ["soundtrack", "orchestral", "epic", "musical"],
    "melancholy": ["sad", "melancholy", "indie folk", "emo"],
    "euphoric": ["pop", "feel good", "dance", "synthpop"],
    "cozy": ["folk", "acoustic", "chill", "lo-fi"],
    "intense": ["alternative rock", "rock", "metal", "dark pop"],
    "nostalgic": ["80s", "synthpop", "indie pop", "retro"],
    "romantic": ["love", "romance", "pop", "rnb"],
    "rebellious": ["punk", "alternative", "rock", "grunge"],
    "chill": ["chill", "lo-fi", "ambient", "downtempo"],
    "dark": ["dark pop", "alternative", "gothic", "industrial"],
    "whimsical": ["indie pop", "quirky", "disney", "musical"],
    "adventurous": ["adventure", "soundtrack", "folk rock", "world"],
    "contemplative": ["indie folk", "ambient", "classical", "piano"],
}

# Google Books subject / keyword hints per vibe
VIBE_TO_BOOK_QUERIES: dict[str, list[str]] = {
    "intimate": ["subject:psychological fiction", "subject:literary fiction"],
    "electric": ["subject:thriller", "subject:young adult"],
    "dreamy": ["subject:fantasy", "subject:magical realism"],
    "epic": ["subject:epic fantasy", "subject:mythology", "subject:space opera"],
    "melancholy": ["subject:tragedy", "subject:literary fiction"],
    "romantic": ["subject:romance", "subject:romantic fantasy"],
    "dark": ["subject:horror", "subject:dark fantasy"],
    "adventurous": ["subject:adventure", "subject:quest fantasy"],
    "whimsical": ["subject:humor", "subject:fairy tales"],
    "intense": ["subject:suspense", "subject:thriller"],
    "nostalgic": ["subject:historical fiction", "subject:classics"],
    "cozy": ["subject:cozy mystery", "subject:contemporary romance"],
    "contemplative": ["subject:philosophy", "subject:memoir"],
    "chill": ["subject:slice of life", "subject:contemporary"],
    "euphoric": ["subject:feel good fiction", "subject:romance"],
    "rebellious": ["subject:dystopia", "subject:coming of age"],
}


def _env(name: str) -> str | None:
    v = (os.getenv(name) or "").strip()
    return v or None


def lastfm_tags_for_vibes(vibe_labels: list[str] | str, limit: int = 6) -> list[str]:
    if isinstance(vibe_labels, str):
        vibes = [v.strip().lower() for v in vibe_labels.split(",") if v.strip()]
    else:
        vibes = [v.strip().lower() for v in vibe_labels if v.strip()]
    tags: list[str] = []
    for v in vibes:
        for t in VIBE_TO_LASTFM_TAGS.get(v, []):
            if t not in tags:
                tags.append(t)
            if len(tags) >= limit:
                return tags
    if not tags:
        tags = ["pop", "indie", "alternative"]
    return tags[:limit]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def _lastfm_get(client: httpx.AsyncClient, params: dict) -> dict:
    key = _env("LASTFM_API_KEY")
    if not key:
        return {}
    params = {**params, "api_key": key, "format": "json"}
    r = await client.get(LASTFM, params=params)
    if r.status_code != 200:
        return {}
    return r.json()


async def fetch_tracks_for_tag(
    client: httpx.AsyncClient, tag: str, limit: int = 20
) -> list[dict[str, str]]:
    data = await _lastfm_get(
        client,
        {"method": "tag.getTopTracks", "tag": tag, "limit": limit},
    )
    tracks = ((data.get("tracks") or {}).get("track")) or []
    if isinstance(tracks, dict):
        tracks = [tracks]
    out = []
    for t in tracks:
        name = t.get("name") or ""
        artist = ((t.get("artist") or {}).get("name")) if isinstance(t.get("artist"), dict) else t.get("artist")
        artist = artist or ""
        if name and artist:
            out.append({"title": name, "creator": artist, "url": t.get("url") or ""})
    return out


async def fetch_track_tags(
    client: httpx.AsyncClient, title: str, artist: str
) -> list[str]:
    data = await _lastfm_get(
        client,
        {
            "method": "track.getTopTags",
            "track": title,
            "artist": artist,
            "autocorrect": 1,
        },
    )
    tags_raw = ((data.get("toptags") or {}).get("tag")) or []
    if isinstance(tags_raw, dict):
        tags_raw = [tags_raw]
    tags = [t.get("name", "").lower() for t in tags_raw if t.get("name")]
    if len(tags) >= 3:
        return tags[:20]

    adata = await _lastfm_get(
        client,
        {"method": "artist.getTopTags", "artist": artist, "autocorrect": 1},
    )
    at = ((adata.get("toptags") or {}).get("tag")) or []
    if isinstance(at, dict):
        at = [at]
    tags.extend(t.get("name", "").lower() for t in at if t.get("name"))
    return tags[:20]


async def discover_songs_by_vibes(
    vibe_labels: list[str] | str,
    query_vector: list[float] | np.ndarray,
    top_n: int = 8,
    per_tag: int = 15,
) -> list[dict[str, Any]]:
    """Pull current Last.fm tracks for vibe-related tags and rank by vibe cosine."""
    if not _env("LASTFM_API_KEY"):
        return []

    tags = lastfm_tags_for_vibes(vibe_labels)
    qv = np.asarray(query_vector, dtype=np.float32)

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Gather candidate tracks across tags
        batches = await asyncio.gather(
            *[fetch_tracks_for_tag(client, tag, limit=per_tag) for tag in tags]
        )
        seen: set[tuple[str, str]] = set()
        candidates: list[dict[str, str]] = []
        for batch in batches:
            for tr in batch:
                key = (tr["title"].lower(), tr["creator"].lower())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(tr)

        # Cap tag-fetch work
        candidates = candidates[:60]

        async def score_one(tr: dict[str, str]) -> dict[str, Any] | None:
            try:
                tags_list = await fetch_track_tags(client, tr["title"], tr["creator"])
            except Exception:
                return None
            cleaned = clean_tags(tags_list)
            text = f"{tr['title']} {tr['creator']} {' '.join(cleaned)}"
            vec = tags_to_vibe_vector(cleaned, text=text)
            sim = _cosine(qv, np.asarray(vec, dtype=np.float32))
            if sim < 0.2:
                return None
            primary = primary_vibe(vec)
            labels = ", ".join(top_vibes(vec, 4))
            return {
                "title": tr["title"],
                "creator": tr["creator"],
                "type": "song",
                "similarity": round(sim, 4),
                "primary_vibe": primary,
                "vibe_labels": labels,
                "why": f"Similar mood: {labels or primary}.",
                "source": "live",
                "url": tr.get("url") or "",
                "tags": cleaned[:15],
            }

        scored = await asyncio.gather(*[score_one(tr) for tr in candidates])

    results = [s for s in scored if s]
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]


async def discover_books_by_vibes(
    vibe_labels: list[str] | str,
    query_vector: list[float] | np.ndarray,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """Pull books from Google Books using vibe-related subjects and rank by vibe."""
    key = _env("GOOGLE_BOOKS_API_KEY")
    if isinstance(vibe_labels, str):
        vibes = [v.strip().lower() for v in vibe_labels.split(",") if v.strip()]
    else:
        vibes = [v.strip().lower() for v in vibe_labels if v.strip()]

    queries: list[str] = []
    for v in vibes[:4]:
        for q in VIBE_TO_BOOK_QUERIES.get(v, []):
            if q not in queries:
                queries.append(q)
    if not queries:
        queries = ["subject:fiction"]

    qv = np.asarray(query_vector, dtype=np.float32)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=20.0) as client:
        for q in queries[:4]:
            params: dict[str, Any] = {
                "q": q,
                "maxResults": 10,
                "printType": "books",
                "orderBy": "relevance",
            }
            if key:
                params["key"] = key
            try:
                r = await client.get(GOOGLE_BOOKS, params=params)
                if r.status_code != 200:
                    continue
                items = r.json().get("items") or []
            except Exception:
                continue

            for it in items:
                info = it.get("volumeInfo") or {}
                title = info.get("title") or ""
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                authors = info.get("authors") or ["Unknown"]
                description = info.get("description") or ""
                categories = info.get("categories") or []
                tags = []
                for c in categories:
                    tags.extend(re.split(r"[,/]", c))
                for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", description.lower()):
                    if word in {
                        "romance",
                        "fantasy",
                        "horror",
                        "thriller",
                        "mystery",
                        "adventure",
                        "mythology",
                        "dragons",
                        "magic",
                        "dystopian",
                        "romantic",
                        "epic",
                        "love",
                    }:
                        tags.append(word)
                cleaned = clean_tags(tags)
                text = f"{title} {' '.join(authors)} {description} {' '.join(cleaned)}"
                vec = tags_to_vibe_vector(cleaned, text=text)
                sim = _cosine(qv, np.asarray(vec, dtype=np.float32))
                if sim < 0.18:
                    continue
                primary = primary_vibe(vec)
                labels = ", ".join(top_vibes(vec, 4))
                results.append(
                    {
                        "title": title,
                        "creator": ", ".join(authors),
                        "type": "book",
                        "similarity": round(sim, 4),
                        "primary_vibe": primary,
                        "vibe_labels": labels,
                        "why": f"Similar mood: {labels or primary}.",
                        "source": "live",
                        "tags": cleaned[:15],
                    }
                )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]


def vector_from_labels(vibe_labels: str, tags: list[str] | None = None, text: str = "") -> list[float]:
    cleaned = clean_tags(tags or [])
    # boost labeled vibes directly
    vec = tags_to_vibe_vector(cleaned, text=text)
    for label in [v.strip().lower() for v in (vibe_labels or "").split(",") if v.strip()]:
        if label in VIBE_INDEX:
            i = VIBE_INDEX[label]
            vec[i] = min(1.0, vec[i] + 0.85)
    return vec
