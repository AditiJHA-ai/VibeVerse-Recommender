"""
Expand the permanent song (and optionally book) catalog using live APIs.

Usage:
  python -m backend.expand_catalog
  python -m backend.expand_catalog --songs-only --per-tag 25
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"), override=True)

from backend.build_catalog import build  # noqa: E402
from backend.discovery import (  # noqa: E402
    VIBE_TO_LASTFM_TAGS,
    fetch_track_tags,
    fetch_tracks_for_tag,
)
import httpx  # noqa: E402

# Extra modern / useful Last.fm tags beyond the vibe map
EXTRA_SONG_TAGS = [
    "pop",
    "indie pop",
    "singer-songwriter",
    "musical",
    "soundtrack",
    "rnb",
    "k-pop",
    "hyperpop",
    "bedroom pop",
    "folk",
    "alternative",
    "synthpop",
]


async def collect_songs(per_tag: int = 20, max_new: int = 800) -> pd.DataFrame:
    tags: list[str] = []
    for group in VIBE_TO_LASTFM_TAGS.values():
        for t in group:
            if t not in tags:
                tags.append(t)
    for t in EXTRA_SONG_TAGS:
        if t not in tags:
            tags.append(t)

    rows = []
    seen: set[tuple[str, str]] = set()

    async with httpx.AsyncClient(timeout=25.0) as client:
        for tag in tags:
            if len(rows) >= max_new:
                break
            print(f"  tag: {tag} ...")
            try:
                tracks = await fetch_tracks_for_tag(client, tag, limit=per_tag)
            except Exception as e:
                print(f"    skip ({e})")
                continue
            for tr in tracks:
                key = (tr["title"].lower(), tr["creator"].lower())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    tag_list = await fetch_track_tags(client, tr["title"], tr["creator"])
                except Exception:
                    tag_list = [tag]
                # ensure the discovery tag is present
                if tag.lower() not in tag_list:
                    tag_list = [tag.lower()] + tag_list
                rows.append(
                    {
                        "title": tr["title"],
                        "creator": tr["creator"],
                        "type": "song",
                        "tags": " ".join(tag_list[:25]),
                        "features": f"{tr['creator']} {' '.join(tag_list[:25])}",
                    }
                )
                if len(rows) >= max_new:
                    break
            await asyncio.sleep(0.15)  # be polite to Last.fm

    return pd.DataFrame(rows)


def merge_into_main(new_df: pd.DataFrame) -> int:
    main_path = os.path.join(ROOT, "main_dataframe.pkl")
    base = pd.read_pickle(main_path)
    before = len(base)

    # normalize keys for dedupe
    def key_row(r):
        return (str(r["title"]).strip().lower(), str(r["creator"]).strip().lower(), str(r["type"]))

    existing = {key_row(r) for _, r in base.iterrows()}
    add_rows = []
    for _, r in new_df.iterrows():
        k = key_row(r)
        if k in existing:
            continue
        existing.add(k)
        add_rows.append(r)

    if not add_rows:
        print("No new rows to add.")
        return 0

    merged = pd.concat([base, pd.DataFrame(add_rows)], ignore_index=True)
    merged.to_pickle(main_path)
    added = len(merged) - before
    print(f"Added {added} items -> {len(merged)} total. Rebuilding vibe catalog...")
    build()
    # invalidate recommender cache if API is running
    try:
        from backend.recommender import clear_recommender_cache

        clear_recommender_cache()
    except Exception:
        pass
    return added


async def main_async(args: argparse.Namespace) -> None:
    if not os.getenv("LASTFM_API_KEY"):
        raise SystemExit("LASTFM_API_KEY missing in .env")

    print("Collecting songs from Last.fm...")
    songs = await collect_songs(per_tag=args.per_tag, max_new=args.max_new)
    print(f"Collected {len(songs)} unique tracks.")
    if songs.empty:
        return
    # save a snapshot of the expansion batch
    out = os.path.join(ROOT, "data", "expansion_songs.parquet")
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    try:
        songs.to_parquet(out, index=False)
    except Exception:
        songs.to_pickle(out.replace(".parquet", ".pkl"))
    merge_into_main(songs)


def main() -> None:
    p = argparse.ArgumentParser(description="Expand VibeVerse catalog via Last.fm")
    p.add_argument("--per-tag", type=int, default=20)
    p.add_argument("--max-new", type=int, default=600)
    p.add_argument("--songs-only", action="store_true", default=True)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
