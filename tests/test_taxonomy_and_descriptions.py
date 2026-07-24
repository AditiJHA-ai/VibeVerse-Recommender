"""Vibe taxonomy + description cleaning unit tests."""

from __future__ import annotations

from backend.enrichment import (
    _artists_compatible,
    _clean_description,
    _wiki_belongs_to_artist,
)
from backend.vibe_taxonomy import (
    clean_tags,
    is_noise_tag,
    primary_vibe,
    tags_to_vibe_vector,
)


class TestTagCleaning:
    def test_strips_shelf_noise(self):
        tags = clean_tags(
            "to-read currently-reading fantasy young-adult favorites owned kindle moodromantic"
        )
        assert "to-read" not in tags
        assert "currently-reading" not in tags
        assert "favorites" not in tags
        assert "owned" not in tags
        assert "kindle" not in tags
        joined = " ".join(tags)
        assert "fantasy" in joined or "young-adult" in joined or "moodromantic" in joined

    def test_noise_detector(self):
        assert is_noise_tag("to-read")
        assert is_noise_tag("books-i-own")
        assert not is_noise_tag("fantasy")


class TestVibeVectors:
    def test_song_mood_maps(self):
        vec = tags_to_vibe_vector(["mood_sad", "mood_acoustic"], text="")
        assert primary_vibe(vec) in {"intimate", "melancholy", "cozy", "chill", "dreamy"}
        assert sum(vec) > 0

    def test_book_fantasy_maps(self):
        vec = tags_to_vibe_vector(["fantasy", "young-adult", "moodromantic"], text="Fourth Wing dragons")
        labels_signal = sum(vec) > 0
        assert labels_signal
        assert primary_vibe(vec) in {
            "dreamy",
            "epic",
            "electric",
            "romantic",
            "intimate",
            "adventurous",
            "intense",
        }

    def test_hyphen_tags_not_shredded_away(self):
        # young-adult should still contribute after cleaning/normalize path
        vec = tags_to_vibe_vector(["young-adult", "high-fantasy"], text="")
        assert sum(vec) > 0


class TestDescriptionSafety:
    def test_strips_lastfm_footer(self):
        raw = "A hopeful song about youth. Read more on Last.fm."
        cleaned = _clean_description(raw)
        assert "last.fm" not in cleaned.lower()

    def test_cuts_on_sentence(self):
        raw = "First sentence here. Second sentence here. Third one as well."
        cleaned = _clean_description(raw, max_chars=40)
        assert cleaned.endswith(".") or cleaned.endswith("…")
        assert "Third" not in cleaned or cleaned.endswith(".")

    def test_wiki_rejects_alphaville_on_blackpink(self):
        wiki = (
            "On the surface, this is a hopeful song celebrating youth. "
            "Alphaville was a German Synthpop group. Cold War references."
        )
        assert not _wiki_belongs_to_artist(wiki, "BLACKPINK")

    def test_wiki_accepts_blackpink_blurb(self):
        wiki = (
            "Forever Young is the sub-title track on BLACKPINK's first mini-album "
            "SQUARE UP with a Moombahton rhythm."
        )
        assert _wiki_belongs_to_artist(wiki, "BLACKPINK")

    def test_artists_compatible(self):
        assert _artists_compatible("BLACKPINK", "Blackpink")
        assert _artists_compatible("TOMORROW X TOGETHER", "Tomorrow X Together")
        assert not _artists_compatible("BLACKPINK", "Alphaville")
