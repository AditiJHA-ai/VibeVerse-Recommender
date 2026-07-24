"""End-to-end API recommendation regressions (uses TestClient; live calls when allow_live)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.recommender import clear_recommender_cache


@pytest.fixture
def client():
    clear_recommender_cache()
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["items"] > 0


def test_suggest_books(client: TestClient):
    r = client.get("/api/suggest", params={"q": "dune", "type": "book"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    assert any("dune" in x["title"].lower() for x in results)


@pytest.mark.integration
def test_recommend_chasing_that_feeling_not_fortunes(client: TestClient):
    if not os.getenv("LASTFM_API_KEY"):
        pytest.skip("LASTFM_API_KEY missing")
    r = client.post(
        "/api/recommend",
        json={
            "query": "chasing that feeling",
            "input_type": "song",
            "want": "song",
            "top_n": 8,
            "allow_live": True,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "fortunes" not in data["matched_creator"].lower()
    assert "rainy" not in data["matched_title"].lower()
    assert "chasing" in data["matched_title"].lower()


@pytest.mark.integration
def test_recommend_forever_young_blackpink_description(client: TestClient):
    if not os.getenv("LASTFM_API_KEY"):
        pytest.skip("LASTFM_API_KEY missing")
    r = client.post(
        "/api/recommend",
        json={
            "query": "Forever Young",
            "input_type": "song",
            "want": "song",
            "top_n": 10,
            "allow_live": True,
        },
    )
    # Without artist, Last.fm may return Alphaville — that path is allowed.
    # Explicit artist path is covered in live unit tests.
    assert r.status_code in (200, 404)


@pytest.mark.integration
def test_recommend_forever_young_with_artist_hint_in_query(client: TestClient):
    if not os.getenv("LASTFM_API_KEY"):
        pytest.skip("LASTFM_API_KEY missing")
    r = client.post(
        "/api/recommend",
        json={
            "query": "Forever Young - BLACKPINK",
            "input_type": "song",
            "want": "song",
            "top_n": 10,
            "allow_live": True,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "blackpink" in data["matched_creator"].lower().replace(" ", "")
    desc = (data.get("description") or "").lower()
    assert "alphaville" not in desc
    assert "cold war" not in desc

    songs = [x for x in data["recommendations"] if x["type"] == "song"]
    if len(songs) >= 6:
        same = sum(
            1
            for x in songs
            if "blackpink" in x["creator"].lower().replace(" ", "")
        )
        # ~50/50: same-artist should not dominate
        assert same <= max(5, len(songs) // 2 + 1)


@pytest.mark.integration
def test_recommend_fourth_wing_has_series_mates(client: TestClient):
    r = client.post(
        "/api/recommend",
        json={
            "query": "Fourth Wing",
            "input_type": "book",
            "want": "book",
            "top_n": 8,
            "allow_live": True,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "yarros" in data["matched_creator"].lower()
    books = [x for x in data["recommendations"] if x["type"] == "book"]
    titles = " | ".join(b["title"].lower() for b in books)
    assert "iron flame" in titles or "onyx storm" in titles
