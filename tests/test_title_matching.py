"""Regression tests for title matching bugs we hit in production."""

from __future__ import annotations

import pytest

from backend.api import _query_matches_title
from backend.related import extract_series_hint
from backend.recommender import VibeRecommender


class TestQueryMatchesTitle:
    def test_rejects_feeling_collision(self):
        # "chasing that feeling" must NOT accept The Fortunes track
        assert not _query_matches_title(
            "chasing that feeling",
            "Here Comes That Rainy Day Feeling Again",
        )

    def test_accepts_real_txt_title(self):
        assert _query_matches_title("chasing that feeling", "Chasing That Feeling")

    def test_rejects_ur_inside_fourth(self):
        # Short catalog title "UR" must not match "fourth wing"
        assert not _query_matches_title("fourth wing", "UR")

    def test_accepts_series_wrapper(self):
        assert _query_matches_title("harry potter", "Harry Potter series")

    def test_accepts_substring_real_book(self):
        assert _query_matches_title(
            "dune messiah",
            "Dune Messiah (Dune Chronicles #2)",
        )

    def test_accepts_onyx_storm_exact(self):
        assert _query_matches_title("onyx storm", "Onyx Storm")


class TestCatalogResolveRegressions:
    def test_fourth_wing_does_not_resolve_to_ur(self, recommender: VibeRecommender):
        hit = recommender.resolve_title("fourth wing", prefer_type="book")
        if hit is not None:
            _, title = hit
            assert "ur" != title.strip().lower()
            assert "stephen" not in title.lower()
        # Prefer live for franchise-less modern titles not in catalog
        assert hit is None or "fourth wing" in title.lower()

    def test_chasing_that_feeling_does_not_resolve_to_fortunes(
        self, recommender: VibeRecommender
    ):
        hit = recommender.resolve_title("chasing that feeling", prefer_type="song")
        assert hit is None or "fortunes" not in hit[1].lower()
        assert hit is None or "rainy" not in hit[1].lower()

    def test_harry_potter_does_not_force_goblet(self, recommender: VibeRecommender):
        hit = recommender.resolve_title("harry potter", prefer_type="book")
        # Should defer to live series lookup, not Goblet of Fire
        assert hit is None

    def test_percy_jackson_does_not_force_ultimate_guide(
        self, recommender: VibeRecommender
    ):
        hit = recommender.resolve_title("percy jackson", prefer_type="book")
        assert hit is None

    def test_specific_catalog_book_still_resolves(self, recommender: VibeRecommender):
        hit = recommender.resolve_title("dune messiah", prefer_type="book")
        assert hit is not None
        assert "dune messiah" in hit[1].lower()

    def test_three_dark_crowns_resolves(self, recommender: VibeRecommender):
        hit = recommender.resolve_title("three dark crowns", prefer_type="book")
        assert hit is not None
        assert "three dark crowns" in hit[1].lower()


class TestSeriesHintExtraction:
    def test_paren_series(self):
        assert (
            extract_series_hint(
                "Harry Potter and the Goblet of Fire (Harry Potter, #4)"
            )
            == "Harry Potter"
        )

    def test_series_suffix(self):
        assert extract_series_hint("Percy Jackson series") == "Percy Jackson"

    def test_plain_title_may_be_none(self):
        # Fourth Wing has no series marker in the title string
        assert extract_series_hint("Fourth Wing") in (None, "Fourth Wing")
