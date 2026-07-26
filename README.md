---
title: VibeVerse
emoji: 📚
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Book ↔ song recommender by emotional vibe
---

# VibeVerse

Cross-domain recommender: match books and songs by shared emotional vibe - one site, landing page + explore page.

## Stack

- **Frontend:** React (Vite) — Gamma-style landing + `/explore` app
- **Backend:** FastAPI + vibe taxonomy matching
- **Live lookup:** Open Library (books, free/no key) + Google Books fallback + Last.fm (songs)

## Setup

```bash
# 1. Python deps
python -m pip install -r requirements.txt

# 2. Build vibe catalog (from existing main_dataframe.pkl)
python -m backend.build_catalog

# 3. API keys (required for Fourth Wing / new titles / live song lookup)
copy .env.example .env
# Edit .env and set:
#   GOOGLE_BOOKS_API_KEY=...
#   LASTFM_API_KEY=...

# 4. Run API
python -m backend.api

# 5. In another terminal — frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — landing at `/`, recommender at `/explore`.
API defaults to http://127.0.0.1:8001 (set `PORT` in `.env` if needed).

## How matching works

1. Strip Goodreads shelf noise (`to-read`, `owned`, …)
2. Map book/song tags into one vibe vector (Intimate / Electric / Dreamy / Epic + finer vibes)
3. Cosine-match across domains
4. If the title isn’t in the local catalog, enrich via Google Books / Last.fm
5. **Also discover live recommendations** from Last.fm / Google Books by vibe tags (so results aren’t limited to the old offline song dump)

## Tests

```bash
# Fast offline regressions (title bugs, taxonomy, 50/50 blend)
pytest -m "not integration"

# Full suite including live Open Library / Last.fm checks
pytest
# or only live:
pytest -m integration
```

## Expand the permanent song catalog

Live discovery already surfaces modern tracks at recommend-time. To also grow the offline library:

```bash
python -m backend.expand_catalog --per-tag 20 --max-new 600
```

This pulls Last.fm top tracks for vibe/genre tags, merges them into `main_dataframe.pkl`, and rebuilds `data/`.

## Deploy (free - Hugging Face Spaces)

No Railway/Render trial needed. Hugging Face Spaces is free for public apps.

1. Create a free account: https://huggingface.co/join
2. Create a **Docker** Space named `VibeVerse` (SDK: Docker, port `7860`)
3. In Space **Settings → Secrets**, add:
   - `GOOGLE_BOOKS_API_KEY`
   - `LASTFM_API_KEY`
4. Push this repo to the Space (or connect GitHub):

```bash
hf auth login
hf upload YOUR_HF_USERNAME/VibeVerse . --repo-type=space
```

Your live URL will be: `https://YOUR_HF_USERNAME-VibeVerse.hf.space`

Cold starts after idle are normal on the free tier (first load can take ~30–60s).

## Legacy

`app.py` is the old Streamlit prototype. Prefer the React + FastAPI app above.
