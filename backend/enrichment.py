"""Live enrichment via Open Library + Google Books (fallback) + Last.fm."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
OPEN_LIBRARY = "https://openlibrary.org"
GOOGLE_BOOKS = "https://www.googleapis.com/books/v1/volumes"
LASTFM = "https://ws.audioscrobbler.com/2.0/"

CONTENT_WORDS = {
    "romance",
    "fantasy",
    "horror",
    "thriller",
    "mystery",
    "adventure",
    "mythology",
    "dragons",
    "war",
    "love",
    "magic",
    "dystopian",
    "space",
    "epic",
    "tragic",
    "comedy",
    "memoir",
    "biography",
    "poetry",
    "romantic",
    "suspense",
    "dragon",
    "warriors",
    "academy",
}


def _env(name: str) -> str | None:
    v = os.getenv(name, "").strip()
    return v or None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _clean_description(text: str, max_chars: int = 900) -> str:
    """Keep a readable plot blurb; drop review-quote piles; end on a sentence."""
    raw = _strip_html(text)
    if not raw:
        return ""
    # Last.fm footers / site chrome
    raw = re.sub(r"\s*Read more on Last\.?fm\.?\s*", " ", raw, flags=re.I)
    raw = re.sub(r"\s*<a href=.*$", "", raw, flags=re.I)
    # Normalize whitespace
    raw = re.sub(r"\s+", " ", raw).strip()

    # Drop trailing press-quote walls ("…'Pure escapism' INDEPENDENT …")
    press_wall = re.search(
        r"(?:[.!?]\s+)?[\"'“][^\"'”]{15,}[\"'”]\s+[A-Z][A-Z\s]{2,20}\b",
        raw,
    )
    if press_wall and press_wall.start() > 100:
        raw = raw[: press_wall.start()].strip()

    quote_hits = len(re.findall(r"[“\"'].{10,120}[”\"']", raw))
    if quote_hits >= 3:
        m = re.search(r"[“\"']", raw)
        if m and m.start() > 80:
            raw = raw[: m.start()].strip()

    # Split into sentences and keep whole ones under the budget
    parts = re.split(r"(?<=[.!?])\s+", raw)
    kept: list[str] = []
    total = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # skip tiny review attributions like "THE SUN"
        if len(p) < 28 and p.isupper():
            continue
        if total + len(p) + 1 > max_chars:
            break
        kept.append(p)
        total += len(p) + 1
        if len(kept) >= 4:
            break

    if kept:
        return " ".join(kept)
    # Fallback: cut at last sentence end inside budget
    chunk = raw[:max_chars]
    end = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
    if end > 80:
        return chunk[: end + 1].strip()
    return chunk.rstrip(" ,;:-") + ("…" if len(raw) > max_chars else "")


def _tags_from_text(*parts: str) -> list[str]:
    blob = " ".join(p for p in parts if p).lower()
    tags = []
    for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", blob):
        if word in CONTENT_WORDS:
            tags.append(word)
    return tags


_COMPANION_NOISE = (
    "guide",
    "cookbook",
    "coloring",
    "colouring",
    "journal",
    "trivia",
    "quiz",
    "companion",
    "encyclopedia",
    "workbook",
    "behind the scenes",
    "ultimate guide",
)


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _score_open_library_doc(doc: dict, q: str) -> float:
    title = (doc.get("title") or "").lower()
    score = 0.0
    nq, nt = _norm_title(q), _norm_title(title)
    if nt == nq:
        score += 140
    elif nt.startswith(nq) and len(nt) - len(nq) < 10:
        score += 90
    elif title.startswith(q):
        score += 70
    elif q in title:
        score += 35
    if any(n in title for n in _COMPANION_NOISE):
        score -= 100
    # Spinoff / secondary books that steal franchise searches
    for bad in (
        "greek gods",
        "greek heroes",
        "schoolbooks",
        "cursed child",
        "illustrated",
        "movie",
        "screenplay",
    ):
        if bad in title:
            score -= 60
    # Prefer first books in well-known series
    for marker in (
        "philosopher",
        "sorcerer",
        "lightning thief",
        "the lightning thief",
        "#1",
        "book 1",
        "book one",
    ):
        if marker in title:
            score += 55
            break
    # Popularity / completeness signals
    score += min(float(doc.get("edition_count") or 0), 80) / 8.0
    if doc.get("cover_i"):
        score += 6
    year = doc.get("first_publish_year")
    if isinstance(year, int) and 1990 <= year <= 2012:
        score += 3
    return score


async def lookup_book_open_library(query: str) -> dict[str, Any] | None:
    """Primary book lookup — free, no API key (Search + Work + Covers)."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        r = await client.get(
            OPEN_LIBRARY_SEARCH,
            params={"q": query, "limit": 12},
            headers={"User-Agent": "VibeVerse/2.0 (book-song recommender)"},
        )
        if r.status_code != 200:
            return None
        docs = r.json().get("docs") or []
        if not docs:
            return None

        q = query.strip().lower()
        ranked = sorted(docs, key=lambda d: _score_open_library_doc(d, q), reverse=True)
        best = ranked[0]

        title = best.get("title") or query
        authors = best.get("author_name") or ["Unknown"]
        subjects = list(best.get("subject") or [])
        first_sentence = best.get("first_sentence")
        if isinstance(first_sentence, list):
            first_sentence = first_sentence[0] if first_sentence else ""
        description = first_sentence or ""

        # Fetch richer work description when available
        work_key = best.get("key")  # e.g. /works/OL123W
        if work_key:
            try:
                wr = await client.get(
                    f"{OPEN_LIBRARY}{work_key}.json",
                    headers={"User-Agent": "VibeVerse/2.0 (book-song recommender)"},
                )
                if wr.status_code == 200:
                    work = wr.json()
                    desc = work.get("description")
                    if isinstance(desc, dict):
                        description = desc.get("value") or description
                    elif isinstance(desc, str) and desc.strip():
                        description = desc
                    for s in work.get("subjects") or []:
                        if isinstance(s, str):
                            subjects.append(s)
            except Exception:
                pass

        related = sum(1 for d in docs if q in (d.get("title") or "").lower())
        nq, nt = _norm_title(q), _norm_title(title)
        # Exact/near-exact book title (e.g. "onyx storm") is NOT a franchise query
        specific_book = nt == nq or (nt.startswith(nq) and len(nt) - len(nq) <= 8)
        franchise_query = (
            related >= 4 and len(q.split()) <= 3 and not specific_book
        )
        display_title = (
            f"{query.strip().title()} series" if franchise_query else title
        )

        cover_id = best.get("cover_i")
        thumbnail = (
            f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None
        )

        tags = [str(s).lower() for s in subjects[:20]]
        tags.extend(_tags_from_text(description, title, query))
        if franchise_query:
            tags.extend(["fantasy", "adventure", "magic", "young-adult"])

        clean_desc = _clean_description(description)
        if franchise_query:
            series_name = query.strip().title()
            if clean_desc:
                if series_name.lower() not in clean_desc.lower()[:80]:
                    clean_desc = f"The {series_name} series. {clean_desc}"
            else:
                clean_desc = (
                    f"The {series_name} series — a sweeping story world across multiple books, "
                    f"full of adventure, friendship, and wonder."
                )

        return {
            "title": display_title,
            "creator": ", ".join(authors[:3]),
            "type": "book",
            "tags": [t.strip() for t in tags if t and t.strip()],
            "description": clean_desc,
            "thumbnail": thumbnail,
            "info_link": f"https://openlibrary.org{work_key}" if work_key else None,
        }


async def lookup_book_google(query: str, api_key: str | None = None) -> dict[str, Any] | None:
    key = api_key or _env("GOOGLE_BOOKS_API_KEY")
    params: dict[str, Any] = {"q": query, "maxResults": 5, "printType": "books"}
    if key:
        params["key"] = key

    async with httpx.AsyncClient(timeout=12.0) as client:
        r = await client.get(GOOGLE_BOOKS, params=params)
        r.raise_for_status()
        data = r.json()

    items = data.get("items") or []
    if not items:
        return None

    q = query.strip().lower()
    best = items[0]
    for it in items:
        info = it.get("volumeInfo") or {}
        t = (info.get("title") or "").lower()
        if q in t or t in q:
            best = it
            break

    info = best.get("volumeInfo") or {}
    title = info.get("title") or query
    authors = info.get("authors") or ["Unknown"]
    description = info.get("description") or ""
    categories = info.get("categories") or []

    tags: list[str] = []
    for c in categories:
        tags.extend(re.split(r"[,/]", c))
    tags.extend(_tags_from_text(description))

    return {
        "title": title,
        "creator": ", ".join(authors),
        "type": "book",
        "tags": [t.strip().lower() for t in tags if t.strip()],
        "description": _clean_description(description),
        "thumbnail": (info.get("imageLinks") or {}).get("thumbnail"),
        "info_link": info.get("infoLink"),
    }


def _finalize_book_hit(query: str, hit: dict[str, Any], *, franchise: bool = False) -> dict[str, Any]:
    """Normalize title/description for user-facing display."""
    q = query.strip()
    title = hit.get("title") or q
    nq, nt = _norm_title(q), _norm_title(title)
    specific = nt == nq or (nt.startswith(nq) and len(nt) - len(nq) <= 8)
    as_series = franchise and len(q.split()) <= 3 and not specific

    if as_series:
        hit["title"] = f"{q.title()} series"
        desc = (hit.get("description") or "").strip()
        if desc and q.lower() not in desc.lower()[:80]:
            hit["description"] = f"The {q.title()} series. {desc}"
        elif not desc:
            hit["description"] = (
                f"The {q.title()} series — a sweeping story world across multiple books, "
                f"full of adventure, friendship, and wonder."
            )
        tags = list(hit.get("tags") or [])
        tags.extend(["fantasy", "adventure", "magic", "young-adult"])
        hit["tags"] = tags

    hit["description"] = _clean_description(hit.get("description") or "")
    return hit


async def lookup_book(query: str, api_key: str | None = None) -> dict[str, Any] | None:
    # Open Library first (free, great descriptions/covers), Google as fallback
    hit = None
    franchise = False
    try:
        hit = await lookup_book_open_library(query)
        if hit:
            franchise = str(hit.get("title") or "").lower().endswith(" series")
            if hit.get("description") or hit.get("tags"):
                # OL path already sets series title when appropriate
                hit["description"] = _clean_description(hit.get("description") or "")
                return hit
    except Exception:
        hit = None

    try:
        g = await lookup_book_google(query, api_key=api_key)
        if not g:
            return hit
        # Treat short multi-hit style queries as series when Google title isn't exact
        q = query.strip().lower()
        title = (g.get("title") or "").lower()
        nq, nt = _norm_title(q), _norm_title(title)
        specific = nt == nq or (nt.startswith(nq) and len(nt) - len(nq) <= 8)
        looks_franchise = len(q.split()) <= 3 and not specific and q in title
        return _finalize_book_hit(query, g, franchise=looks_franchise or franchise)
    except Exception:
        return hit


def _artists_compatible(expected: str, got: str) -> bool:
    if not expected or not got:
        return True
    a, b = _norm_title(expected), _norm_title(got)
    return a == b or a in b or b in a


def _wiki_belongs_to_artist(wiki: str, artist: str) -> bool:
    """Reject cross-attached bios (e.g. Alphaville wiki on a BLACKPINK track)."""
    if not wiki or not artist:
        return bool(wiki)
    w = wiki.lower()
    a = artist.lower().strip()
    # Positive: artist name appears in the blurb
    tokens = [t for t in re.split(r"\s+", a) if len(t) > 2]
    if a in w or _norm_title(a) in _norm_title(w):
        return True
    if tokens and all(t in w for t in tokens):
        return True
    # Negative: blurb clearly about a different named act
    m = re.search(
        r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3})\s+(?:was|were|is|are)\s+a\b",
        wiki,
    )
    if m:
        mentioned = m.group(1).strip()
        if not _artists_compatible(artist, mentioned):
            return False
    # Cold-war / Alphaville-specific fingerprints when artist isn't Alphaville
    if "alphaville" in w and "alphaville" not in a:
        return False
    if "cold war" in w and "blackpink" in a.replace(" ", ""):
        return False
    return True


def _fallback_song_description(title: str, artist: str, tags: list[str]) -> str:
    moodish = [
        t
        for t in tags
        if t
        and not t.startswith("http")
        and t
        not in {
            "seen live",
            "favorites",
            "favourite",
            "beautiful",
            "love",
            "awesome",
        }
    ][:5]
    if moodish:
        return (
            f"“{title}” by {artist} — a track often tagged "
            f"{', '.join(moodish)}."
        )
    return f"“{title}” by {artist}."


async def lookup_song(
    query: str,
    artist: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    key = api_key or _env("LASTFM_API_KEY")
    if not key:
        return None

    async with httpx.AsyncClient(timeout=12.0) as client:
        track = query
        art = (artist or "").strip()
        if not art:
            for sep in [" - ", " – ", " by "]:
                if sep in query.lower():
                    idx = query.lower().find(sep)
                    track = query[:idx].strip()
                    art = query[idx + len(sep) :].strip()
                    break

        requested_artist = art

        if not art:
            sr = await client.get(
                LASTFM,
                params={
                    "method": "track.search",
                    "track": query,
                    "api_key": key,
                    "format": "json",
                    "limit": 10,
                },
            )
            sr.raise_for_status()
            matches = (
                ((sr.json().get("results") or {}).get("trackmatches") or {}).get("track")
                or []
            )
            if not matches:
                return None
            if isinstance(matches, dict):
                matches = [matches]
            # Prefer exact / near-exact title, then popularity.
            # Avoid picking an old song that only shares a generic word.
            qn = _norm_title(query)
            q_content = {
                w
                for w in re.findall(r"[a-z0-9]{4,}", query.lower())
                if w
                not in {
                    "that",
                    "this",
                    "with",
                    "from",
                    "your",
                    "come",
                    "comes",
                    "again",
                    "feeling",
                    "love",
                    "baby",
                    "girl",
                    "time",
                    "life",
                    "song",
                    "night",
                    "heart",
                    "forever",
                    "young",
                }
                and len(w) >= 5
            }

            def rank_match(m: dict) -> tuple:
                name = (m.get("name") or "").lower()
                nn = _norm_title(name)
                exact = 0 if nn == qn else 1
                contains = 0 if qn in nn or nn in qn else 1
                name_words = set(re.findall(r"[a-z0-9]{4,}", name))
                missing = len(q_content - name_words)
                pop = -int(m.get("listeners") or m.get("playcount") or 0)
                return (exact, contains, missing, pop)

            ranked = sorted(matches, key=rank_match)
            top = ranked[0]
            track = top.get("name") or query
            art = top.get("artist") or ""

        info_r = await client.get(
            LASTFM,
            params={
                "method": "track.getInfo",
                "track": track,
                "artist": art,
                "api_key": key,
                "format": "json",
                "autocorrect": 1,
            },
        )
        info_json = info_r.json() if info_r.status_code == 200 else {}
        track_info = info_json.get("track") or {}
        final_title = track_info.get("name") or track
        artist_obj = track_info.get("artist") or {}
        final_artist = (
            artist_obj.get("name") if isinstance(artist_obj, dict) else art
        ) or art

        # If caller insisted on an artist, never keep a mismatched hit
        if requested_artist and not _artists_compatible(requested_artist, final_artist):
            return None

        r = await client.get(
            LASTFM,
            params={
                "method": "track.getTopTags",
                "track": final_title,
                "artist": final_artist,
                "api_key": key,
                "format": "json",
                "autocorrect": 1,
            },
        )
        tag_data = r.json() if r.status_code == 200 else {}

        tags_raw = ((tag_data.get("toptags") or {}).get("tag")) or []
        if isinstance(tags_raw, dict):
            tags_raw = [tags_raw]
        tags = [t.get("name", "").lower() for t in tags_raw if t.get("name")]

        if len(tags) < 3 and final_artist:
            ar = await client.get(
                LASTFM,
                params={
                    "method": "artist.getTopTags",
                    "artist": final_artist,
                    "api_key": key,
                    "format": "json",
                    "autocorrect": 1,
                },
            )
            if ar.status_code == 200:
                at = ((ar.json().get("toptags") or {}).get("tag")) or []
                if isinstance(at, dict):
                    at = [at]
                tags.extend(t.get("name", "").lower() for t in at if t.get("name"))

        wiki = ((track_info.get("wiki") or {}).get("summary")) or ""
        if _wiki_belongs_to_artist(wiki, final_artist):
            description = _clean_description(wiki, max_chars=700)
        else:
            description = ""
        if not description:
            description = _fallback_song_description(final_title, final_artist, tags)

        return {
            "title": final_title,
            "creator": final_artist,
            "type": "song",
            "tags": tags[:25],
            "description": description,
            "url": track_info.get("url"),
        }


async def enrich(
    query: str,
    prefer: str | None = None,
    artist: str | None = None,
) -> dict[str, Any] | None:
    prefer = (prefer or "").lower() or None
    q = query.strip()
    if not q:
        return None

    looks_song = bool(re.search(r"\s[-–]\s", q)) or " by " in q.lower() or bool(artist)

    if prefer in ("book", "song"):
        order = [prefer]
    elif looks_song:
        order = ["song", "book"]
    else:
        order = ["book", "song"]

    for kind in order:
        try:
            if kind == "book":
                hit = await lookup_book(q)
            else:
                hit = await lookup_song(q, artist=artist)
            if hit and (hit.get("tags") or hit.get("description") or hit.get("title")):
                return hit
        except Exception:
            continue
    return None
