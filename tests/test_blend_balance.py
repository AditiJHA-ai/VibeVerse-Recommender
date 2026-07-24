"""Tests for 50/50 affinity vs diverse vibe blending."""

from __future__ import annotations

from backend.related import _blend_balanced, _same_creator


def _song(title: str, artist: str, sim: float) -> dict:
    return {
        "title": title,
        "creator": artist,
        "type": "song",
        "similarity": sim,
        "primary_vibe": "electric",
        "vibe_labels": "electric, euphoric",
        "why": "Similar mood: electric.",
        "source": "live",
    }


class TestBlendBalance:
    def test_same_creator_helper(self):
        assert _same_creator("BLACKPINK", "blackpink")
        assert not _same_creator("BLACKPINK", "Alphaville")
        assert not _same_creator("BLACKPINK", "The Fortunes")

    def test_song_blend_is_about_half_same_artist(self):
        affinity = [
            _song("How You Like That", "BLACKPINK", 0.95),
            _song("Kill This Love", "BLACKPINK", 0.94),
            _song("DDU-DU DDU-DU", "BLACKPINK", 0.93),
            _song("BOOMBAYAH", "BLACKPINK", 0.92),
            _song("Shut Down", "BLACKPINK", 0.91),
            _song("Pink Venom", "BLACKPINK", 0.90),
            _song("FLOWER", "JISOO", 0.88),
            _song("ROCKSTAR", "LISA", 0.87),
        ]
        others = [
            _song("Dynamite", "BTS", 0.80),
            _song("Fancy", "TWICE", 0.78),
            _song("Next Level", "aespa", 0.76),
            _song("Fearless", "LE SSERAFIM", 0.75),
            _song("Super Shy", "NewJeans", 0.74),
            _song("Hype Boy", "NewJeans", 0.73),
        ]
        out = _blend_balanced(
            affinity,
            others,
            top_n=10,
            anchor_creator="BLACKPINK",
            matched_title="Forever Young",
            item_type="song",
        )
        assert len(out) == 10
        same = [x for x in out if _same_creator(x["creator"], "BLACKPINK")]
        other = [x for x in out if not _same_creator(x["creator"], "BLACKPINK")]
        # ~50/50 with small tolerance
        assert len(same) <= 5
        assert len(other) >= 5
        # Must not be 80%+ same artist
        assert len(same) / len(out) <= 0.55

    def test_excludes_matched_title(self):
        affinity = [_song("Forever Young", "BLACKPINK", 0.99)]
        others = [_song("Dynamite", "BTS", 0.8)]
        out = _blend_balanced(
            affinity,
            others,
            top_n=4,
            anchor_creator="BLACKPINK",
            matched_title="Forever Young",
            item_type="song",
        )
        titles = {x["title"].lower() for x in out}
        assert "forever young" not in titles

    def test_book_blend_keeps_series_mates(self):
        affinity = [
            {
                "title": "Iron Flame",
                "creator": "Rebecca Yarros",
                "type": "book",
                "similarity": 0.93,
            },
            {
                "title": "Onyx Storm",
                "creator": "Rebecca Yarros",
                "type": "book",
                "similarity": 0.92,
            },
        ]
        others = [
            {
                "title": "The Shadow of the Wind",
                "creator": "Carlos Ruiz Zafón",
                "type": "book",
                "similarity": 0.7,
            },
            {
                "title": "Dune",
                "creator": "Frank Herbert",
                "type": "book",
                "similarity": 0.65,
            },
        ]
        out = _blend_balanced(
            affinity,
            others,
            top_n=4,
            anchor_creator="Rebecca Yarros",
            matched_title="Fourth Wing",
            item_type="book",
        )
        titles = {x["title"] for x in out}
        assert "Iron Flame" in titles or "Onyx Storm" in titles
        assert len(out) == 4
