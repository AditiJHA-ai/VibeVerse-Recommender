"""
Live integration tests — network + API keys required.

Run:
  pytest -m integration
Skip offline:
  pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from backend.enrichment import lookup_book, lookup_song
from backend.related import affinity_recommendations, related_books, related_songs


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fourth_wing_is_rebecca_yarros():
    hit = await lookup_book("Fourth Wing")
    assert hit is not None
    assert "fourth wing" in hit["title"].lower() or "empyrean" in hit["title"].lower()
    assert "yarros" in hit["creator"].lower()
    assert hit.get("description")
    assert "alphaville" not in (hit.get("description") or "").lower()


@pytest.mark.asyncio
async def test_onyx_storm_is_single_book_not_series_label():
    hit = await lookup_book("Onyx Storm")
    assert hit is not None
    assert not hit["title"].lower().endswith(" series")
    assert "onyx storm" in hit["title"].lower()


@pytest.mark.asyncio
async def test_harry_potter_franchise_resolves_as_series_or_core():
    hit = await lookup_book("harry potter")
    assert hit is not None
    title = hit["title"].lower()
    assert "harry potter" in title
    # Should not be a random mid-series latch without series framing or book 1
    assert "ultimate guide" not in title


@pytest.mark.asyncio
async def test_forever_young_with_artist_is_blackpink(has_lastfm):
    if not has_lastfm:
        pytest.skip("LASTFM_API_KEY missing")
    hit = await lookup_song("Forever Young", artist="BLACKPINK")
    assert hit is not None
    assert "blackpink" in hit["creator"].lower().replace(" ", "")
    desc = (hit.get("description") or "").lower()
    assert "alphaville" not in desc
    assert "cold war" not in desc


@pytest.mark.asyncio
async def test_chasing_that_feeling_is_txt(has_lastfm):
    if not has_lastfm:
        pytest.skip("LASTFM_API_KEY missing")
    hit = await lookup_song("chasing that feeling")
    assert hit is not None
    creator = hit["creator"].lower()
    assert (
        "tomorrow x together" in creator
        or "txt" in creator.replace(" ", "")
        or "투모로우바이투게더" in creator
    )
    assert "fortunes" not in creator
    assert "rainy" not in hit["title"].lower()


@pytest.mark.asyncio
async def test_fourth_wing_affinity_includes_iron_flame():
    rows = await affinity_recommendations(
        {
            "matched_type": "book",
            "matched_title": "Fourth Wing",
            "matched_creator": "Rebecca Yarros",
            "vibe_labels": "electric, dreamy, epic, romantic",
            "primary_vibe": "electric",
            "live_tags": ["fantasy", "romance", "dragons"],
            "description": "Violet Sorrengail at Basgiath War College",
        }
    )
    titles = " | ".join(r["title"].lower() for r in rows)
    assert "iron flame" in titles or "onyx storm" in titles


@pytest.mark.asyncio
async def test_blackpink_affinity_has_group_and_related(has_lastfm):
    if not has_lastfm:
        pytest.skip("LASTFM_API_KEY missing")
    bundle = await related_songs(title="Pink Venom", artist="BLACKPINK", limit=10)
    assert bundle["items"]
    creators = {i["creator"].lower() for i in bundle["items"]}
    assert any("blackpink" in c.replace(" ", "") for c in creators)
    # Member / related often appear
    joined = " ".join(creators)
    assert any(
        name in joined
        for name in ("jisoo", "lisa", "rosé", "rose", "jennie", "bts", "twice", "aespa")
    ) or len(bundle["items"]) >= 4


@pytest.mark.asyncio
async def test_harry_potter_related_books_are_series_mates():
    bundle = await related_books(
        title="Harry Potter series", author="J. K. Rowling", limit=6
    )
    titles = " | ".join(i["title"].lower() for i in bundle["items"])
    assert "philosopher" in titles or "sorcerer" in titles or "chamber" in titles
    assert "ultimate guide" not in titles
