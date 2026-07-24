"""Related items: other books in a series, more songs by an artist (+ related artists)."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
LASTFM = "https://ws.audioscrobbler.com/2.0/"
GOOGLE_BOOKS = "https://www.googleapis.com/books/v1/volumes"

_UA = {"User-Agent": "VibeVerse/2.0 (book-song recommender)"}


def _env(name: str) -> str | None:
    v = (os.getenv(name) or "").strip()
    return v or None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def extract_series_hint(title: str) -> str | None:
    """
    Pull a series name from common title patterns.
    e.g. 'Harry Potter and the Goblet of Fire (Harry Potter, #4)' -> 'Harry Potter'
         'Onyx Storm' / Empyrean — may return None (caller can use author search)
         'Percy Jackson series' -> 'Percy Jackson'
    """
    t = (title or "").strip()
    if not t:
        return None
    if t.lower().endswith(" series"):
        return t[: -len(" series")].strip()

    # (Series Name, #3) or (Series Name #3)
    m = re.search(r"\(([^)#]+?)\s*,?\s*#\s*\d+\)", t)
    if m:
        return m.group(1).strip()

    # Title: Subtitle patterns are weak; prefer left side before colon only if short
    if ":" in t and len(t.split(":")[0].split()) <= 4:
        left = t.split(":", 1)[0].strip()
        if left.lower() not in {"the", "a", "an"}:
            return left

    # "Harry Potter and the ..." -> Harry Potter
    m = re.match(r"^((?:[A-Z][\w']+\s+){1,3}[A-Z][\w']+)\s+and\s+the\s+", t)
    if m:
        return m.group(1).strip()

    return None


async def related_books(
    *,
    title: str,
    author: str,
    limit: int = 8,
) -> dict[str, Any]:
    """Find other books in the same series / by the same author."""
    series = extract_series_hint(title)
    exclude = {_norm(title)}
    # also exclude "X series" wrapper
    if title.lower().endswith(" series"):
        exclude.add(_norm(title[: -len(" series")]))

    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    queries: list[str] = []
    if series:
        queries.append(series)
        if author and author.lower() not in {"unknown", ""}:
            queries.append(f"{series} {author.split(',')[0]}")
    if author and author.lower() not in {"unknown", ""}:
        queries.append(f'author:"{author.split(",")[0].strip()}"')

    async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
        for q in queries:
            if len(items) >= limit:
                break
            try:
                r = await client.get(
                    OPEN_LIBRARY_SEARCH,
                    params={"q": q, "limit": 20},
                    headers=_UA,
                )
                if r.status_code != 200:
                    continue
                docs = r.json().get("docs") or []
            except Exception:
                continue

            for doc in docs:
                t = (doc.get("title") or "").strip()
                if not t or _norm(t) in exclude or _norm(t) in seen:
                    continue
                # Keep series-ish or same-author results
                authors = doc.get("author_name") or []
                author_ok = True
                if author and authors:
                    a0 = author.split(",")[0].strip().lower()
                    author_ok = any(a0 in (a or "").lower() or (a or "").lower() in a0 for a in authors)
                series_ok = True
                if series:
                    series_ok = _norm(series) in _norm(t) or any(
                        _norm(series) in _norm(s)
                        for s in (doc.get("subject") or [])[:30]
                        if isinstance(s, str)
                    )
                if not (author_ok or series_ok):
                    continue
                # Skip pure guides/trivia
                low = t.lower()
                if any(x in low for x in ("guide", "trivia", "cookbook", "coloring", "journal", "quiz")):
                    continue

                seen.add(_norm(t))
                cover_id = doc.get("cover_i")
                items.append(
                    {
                        "title": t,
                        "creator": ", ".join(authors[:2]) if authors else author,
                        "type": "book",
                        "relation": "series" if series and series_ok else "author",
                        "thumbnail": (
                            f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
                            if cover_id
                            else None
                        ),
                    }
                )
                if len(items) >= limit:
                    break

        # Google Books author fallback if still thin
        if len(items) < 3 and author and _env("GOOGLE_BOOKS_API_KEY"):
            try:
                params = {
                    "q": f'inauthor:"{author.split(",")[0].strip()}"',
                    "maxResults": 12,
                    "printType": "books",
                    "key": _env("GOOGLE_BOOKS_API_KEY"),
                }
                r = await client.get(GOOGLE_BOOKS, params=params)
                if r.status_code == 200:
                    for it in r.json().get("items") or []:
                        info = it.get("volumeInfo") or {}
                        t = (info.get("title") or "").strip()
                        if not t or _norm(t) in exclude or _norm(t) in seen:
                            continue
                        creators = ", ".join(info.get("authors") or [author])
                        seen.add(_norm(t))
                        items.append(
                            {
                                "title": t,
                                "creator": creators,
                                "type": "book",
                                "relation": "author",
                                "thumbnail": (info.get("imageLinks") or {}).get("thumbnail"),
                            }
                        )
                        if len(items) >= limit:
                            break
            except Exception:
                pass

    label = f"More in the {series} series" if series else f"More from {author.split(',')[0].strip()}"
    return {
        "kind": "series" if series else "author",
        "label": label,
        "anchor": series or author,
        "items": items[:limit],
    }


async def _lastfm(client: httpx.AsyncClient, params: dict) -> dict:
    key = _env("LASTFM_API_KEY")
    if not key:
        return {}
    params = {**params, "api_key": key, "format": "json"}
    r = await client.get(LASTFM, params=params)
    if r.status_code != 200:
        return {}
    return r.json()


async def related_songs(
    *,
    title: str,
    artist: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    More tracks by the same artist, plus tracks from similar/related artists
    (covers group → member solos, e.g. BLACKPINK → Jennie/Lisa/Rosé/Jisoo).
    """
    artist = (artist or "").strip()
    if not artist or not _env("LASTFM_API_KEY"):
        return {"kind": "artist", "label": "More from this artist", "anchor": artist, "items": []}

    exclude_track = _norm(title)
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    async with httpx.AsyncClient(timeout=20.0) as client:
        # 1) Same artist top tracks
        data = await _lastfm(
            client,
            {"method": "artist.getTopTracks", "artist": artist, "limit": 12, "autocorrect": 1},
        )
        tracks = ((data.get("toptracks") or {}).get("track")) or []
        if isinstance(tracks, dict):
            tracks = [tracks]
        for tr in tracks:
            name = (tr.get("name") or "").strip()
            art = artist
            if isinstance(tr.get("artist"), dict):
                art = tr["artist"].get("name") or artist
            key = (_norm(name), _norm(art))
            if not name or _norm(name) == exclude_track or key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "title": name,
                    "creator": art,
                    "type": "song",
                    "relation": "artist",
                    "url": tr.get("url") or "",
                }
            )

        # 2) Similar artists (often group members / solo projects)
        sim = await _lastfm(
            client,
            {"method": "artist.getSimilar", "artist": artist, "limit": 8, "autocorrect": 1},
        )
        similar = ((sim.get("similarartists") or {}).get("artist")) or []
        if isinstance(similar, dict):
            similar = [similar]

        related_artists: list[str] = []
        for a in similar:
            name = (a.get("name") or "").strip()
            if name and _norm(name) != _norm(artist):
                related_artists.append(name)

        for rel_artist in related_artists[:6]:
            if len(items) >= limit + 8:
                break
            td = await _lastfm(
                client,
                {
                    "method": "artist.getTopTracks",
                    "artist": rel_artist,
                    "limit": 4,
                    "autocorrect": 1,
                },
            )
            rtracks = ((td.get("toptracks") or {}).get("track")) or []
            if isinstance(rtracks, dict):
                rtracks = [rtracks]
            for tr in rtracks[:3]:
                name = (tr.get("name") or "").strip()
                key = (_norm(name), _norm(rel_artist))
                if not name or key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "title": name,
                        "creator": rel_artist,
                        "type": "song",
                        "relation": "related_artist",
                        "url": tr.get("url") or "",
                    }
                )

    same = [x for x in items if x.get("relation") == "artist"]
    related = [x for x in items if x.get("relation") == "related_artist"]
    ordered = same[: max(5, limit // 2)] + related
    return {
        "kind": "artist",
        "label": "",
        "anchor": artist,
        "items": ordered[:limit],
    }


async def _score_song_candidate(
    client: httpx.AsyncClient,
    item: dict[str, Any],
    query_vec: list[float],
) -> dict[str, Any] | None:
    """Score a song by real vibe overlap with the query track."""
    from backend.discovery import fetch_track_tags
    from backend.vibe_taxonomy import (
        clean_tags,
        primary_vibe,
        tags_to_vibe_vector,
        top_vibes,
    )
    import numpy as np

    try:
        tags = await fetch_track_tags(client, item["title"], item["creator"])
    except Exception:
        tags = []
    cleaned = clean_tags(tags)
    text = f"{item['title']} {item['creator']} {' '.join(cleaned)}"
    vec = tags_to_vibe_vector(cleaned, text=text)
    q = np.asarray(query_vec, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    nq, nv = np.linalg.norm(q), np.linalg.norm(v)
    sim = float(np.dot(q, v) / (nq * nv)) if nq and nv else 0.0

    # Same artist / series-world boost — still requires some vibe signal
    rel = item.get("relation")
    if rel == "artist":
        sim = max(sim, 0.72) + 0.08
    elif rel == "related_artist":
        sim = sim + 0.05
    sim = min(sim, 0.98)
    if sim < 0.35:
        return None

    labels = ", ".join(top_vibes(vec, 4)) or primary_vibe(vec)
    return {
        "title": item["title"],
        "creator": item["creator"],
        "type": "song",
        "similarity": round(sim, 4),
        "primary_vibe": primary_vibe(vec),
        "vibe_labels": labels,
        "why": f"Similar mood: {labels}.",
        "source": "live",
        "url": item.get("url") or "",
    }


def _score_book_candidate(item: dict[str, Any], query_vec: list[float], query_labels: str) -> dict[str, Any]:
    """Series/author neighbours share the story-world vibe — rank them highly."""
    from backend.vibe_taxonomy import primary_vibe, top_vibes

    rel = item.get("relation")
    # Same series ≈ nearly the same emotional world
    sim = 0.93 if rel == "series" else 0.86

    vec = [float(x) for x in query_vec]
    labels = ", ".join(top_vibes(vec, 4)) or query_labels or "dreamy"
    primary = primary_vibe(vec)

    return {
        "title": item["title"],
        "creator": item["creator"],
        "type": "book",
        "similarity": round(sim, 4),
        "primary_vibe": primary,
        "vibe_labels": labels,
        "why": f"Similar mood: {labels}.",
        "source": "live",
        "thumbnail": item.get("thumbnail"),
    }


async def affinity_recommendations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Series mates / same-artist / related-artist tracks scored as normal vibe matches
    so they appear in the main recommendation lists (not a separate category).
    """
    from backend.discovery import vector_from_labels

    mtype = (result.get("matched_type") or "").lower()
    title = result.get("matched_title") or ""
    creator = result.get("matched_creator") or ""
    labels = result.get("vibe_labels") or result.get("primary_vibe") or ""
    tags = result.get("live_tags") or []
    text = f"{title} {creator} {result.get('description') or ''}"
    qvec = vector_from_labels(labels, tags=tags, text=text)

    matched_n = _norm(title.replace(" series", ""))
    scored: list[dict[str, Any]] = []

    if mtype == "book":
        bundle = await related_books(title=title, author=creator, limit=10)
        for it in bundle.get("items") or []:
            if _norm(it.get("title") or "") == matched_n:
                continue
            scored.append(_score_book_candidate(it, qvec, labels))

    elif mtype == "song":
        bundle = await related_songs(title=title, artist=creator, limit=14)
        async with httpx.AsyncClient(timeout=20.0) as client:
            for it in bundle.get("items") or []:
                if _norm(it.get("title") or "") == matched_n:
                    continue
                row = await _score_song_candidate(client, it, qvec)
                if row:
                    scored.append(row)

    scored.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return scored


def _same_creator(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _blend_balanced(
    affinity: list[dict[str, Any]],
    others: list[dict[str, Any]],
    *,
    top_n: int,
    anchor_creator: str,
    matched_title: str,
    item_type: str,
) -> list[dict[str, Any]]:
    """
    ~50/50 mix: half close-affinity (same series / same artist),
    half other vibe matches from different creators.
    """
    matched_n = _norm(matched_title.replace(" series", ""))
    half = max(1, top_n // 2)

    def ok(item: dict) -> bool:
        t = str(item.get("title") or "")
        return bool(t) and _norm(t) != matched_n and item.get("type") == item_type

    # Affinity: for songs, only count SAME artist toward the 50% slot
    # (member solos / related artists sit in the "other" half for variety)
    if item_type == "song":
        close = [
            x
            for x in affinity
            if ok(x) and _same_creator(str(x.get("creator") or ""), anchor_creator)
        ]
        related_other = [
            x
            for x in affinity
            if ok(x) and not _same_creator(str(x.get("creator") or ""), anchor_creator)
        ]
    else:
        # Books: series/author mates are the "close" half
        close = [x for x in affinity if ok(x)]
        related_other = []

    close.sort(key=lambda x: float(x.get("similarity") or 0), reverse=True)
    close = close[:half]

    seen = {(str(x.get("title")).lower(), str(x.get("creator")).lower()) for x in close}
    diverse: list[dict[str, Any]] = []

    # Prefer vibe hits from different creators, then related-artist tracks
    pool = [x for x in others if ok(x)] + related_other
    pool.sort(key=lambda x: float(x.get("similarity") or 0), reverse=True)
    for x in pool:
        creator = str(x.get("creator") or "")
        key = (str(x.get("title")).lower(), creator.lower())
        if key in seen:
            continue
        # For songs, keep the diverse half free of the anchor artist
        if item_type == "song" and _same_creator(creator, anchor_creator):
            continue
        seen.add(key)
        diverse.append(x)
        if len(diverse) >= top_n - len(close):
            break

    # If diverse half is short, fill from remaining affinity (related artists)
    if len(diverse) < top_n - len(close):
        for x in affinity:
            if not ok(x):
                continue
            key = (str(x.get("title")).lower(), str(x.get("creator")).lower())
            if key in seen:
                continue
            if item_type == "song" and _same_creator(str(x.get("creator") or ""), anchor_creator):
                continue
            seen.add(key)
            diverse.append(x)
            if len(diverse) >= top_n - len(close):
                break

    blended = close + diverse
    blended.sort(key=lambda x: float(x.get("similarity") or 0), reverse=True)
    return blended[:top_n]


async def merge_affinity_into_recommendations(result: dict[str, Any], want: str, top_n: int) -> dict[str, Any]:
    """Fold series/artist affinity into main recs with a ~50/50 balance."""
    try:
        affinity = await affinity_recommendations(result)
    except Exception:
        affinity = []

    existing = result.get("recommendations") or []
    anchor = str(result.get("matched_creator") or "")
    title = str(result.get("matched_title") or "")

    if want == "all":
        songs = _blend_balanced(
            affinity,
            existing,
            top_n=top_n,
            anchor_creator=anchor,
            matched_title=title,
            item_type="song",
        )
        books = _blend_balanced(
            affinity,
            existing,
            top_n=max(3, top_n // 2),
            anchor_creator=anchor,
            matched_title=title,
            item_type="book",
        )
        result["recommendations"] = songs + books
    elif want == "song":
        result["recommendations"] = _blend_balanced(
            affinity,
            existing,
            top_n=top_n,
            anchor_creator=anchor,
            matched_title=title,
            item_type="song",
        )
    elif want == "book":
        result["recommendations"] = _blend_balanced(
            affinity,
            existing,
            top_n=top_n,
            anchor_creator=anchor,
            matched_title=title,
            item_type="book",
        )
    else:
        result["recommendations"] = existing[:top_n]

    result.pop("related", None)
    return result
