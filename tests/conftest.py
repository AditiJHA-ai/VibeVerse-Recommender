"""Shared fixtures for VibeVerse tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)


@pytest.fixture(autouse=True)
def _clear_recommender_cache():
    from backend.recommender import clear_recommender_cache

    clear_recommender_cache()
    yield
    clear_recommender_cache()


@pytest.fixture(scope="session")
def has_lastfm() -> bool:
    key = (os.getenv("LASTFM_API_KEY") or "").strip().lower()
    return len(key) >= 8 and not key.startswith("your_")


@pytest.fixture(scope="session")
def has_google_books() -> bool:
    key = (os.getenv("GOOGLE_BOOKS_API_KEY") or "").strip().lower()
    return len(key) >= 8 and not key.startswith("your_")


@pytest.fixture(scope="session")
def recommender():
    from backend.recommender import VibeRecommender

    return VibeRecommender()
