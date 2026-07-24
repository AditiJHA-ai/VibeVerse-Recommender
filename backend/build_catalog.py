"""Build cleaned catalog + vibe vectors from main_dataframe.pkl."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running as script from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.vibe_taxonomy import (  # noqa: E402
    VIBES,
    clean_tags,
    primary_vibe,
    tags_to_vibe_vector,
    top_vibes,
)


def build(source_pkl: str | None = None, out_dir: str | None = None) -> dict:
    source_pkl = source_pkl or os.path.join(ROOT, "main_dataframe.pkl")
    out_dir = out_dir or os.path.join(ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_pickle(source_pkl).copy()
    df["title"] = df["title"].astype(str)
    df["creator"] = df["creator"].astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)
    df["type"] = df["type"].astype(str)

    cleaned = []
    vectors = []
    primaries = []
    vibe_labels = []

    for _, row in df.iterrows():
        tags = clean_tags(row["tags"])
        # include title+creator lightly for text hints (e.g. Odyssey)
        text = f"{row['title']} {row['creator']} {' '.join(tags)}"
        vec = tags_to_vibe_vector(tags, text=text)
        cleaned.append(" ".join(tags))
        vectors.append(vec)
        primaries.append(primary_vibe(vec))
        vibe_labels.append(", ".join(top_vibes(vec, 4)))

    df["clean_tags"] = cleaned
    df["primary_vibe"] = primaries
    df["vibe_labels"] = vibe_labels
    # readable creator (songs often lowercase glued; books too)
    df["creator_display"] = df["creator"].str.replace(r"(?<!^)(?=[A-Z])", " ", regex=True)

    matrix = np.asarray(vectors, dtype=np.float32)
    # L2-normalize for cosine via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix_norm = matrix / norms

    # Search index: title lower → list of row indices (handle dupes)
    index: dict[str, list[int]] = {}
    for i, title in enumerate(df["title"].tolist()):
        key = title.strip().lower()
        index.setdefault(key, []).append(i)

    catalog_path = os.path.join(out_dir, "catalog.parquet")
    try:
        df.to_parquet(catalog_path, index=False)
    except Exception:
        catalog_path = os.path.join(out_dir, "catalog.pkl")
        df.to_pickle(catalog_path)

    np.save(os.path.join(out_dir, "vibe_matrix.npy"), matrix_norm)
    with open(os.path.join(out_dir, "search_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)

    meta = {
        "n_items": int(len(df)),
        "n_books": int((df["type"] == "book").sum()),
        "n_songs": int((df["type"] == "song").sum()),
        "vibes": VIBES,
        "catalog": os.path.basename(catalog_path),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Coverage stats
    empty = int((matrix.sum(axis=1) == 0).sum())
    print(f"Built catalog: {meta['n_books']} books, {meta['n_songs']} songs")
    print(f"Items with zero vibe signal: {empty} ({100 * empty / len(df):.1f}%)")
    print(f"Wrote {out_dir}")
    return meta


if __name__ == "__main__":
    build()
