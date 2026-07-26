from backend.showcase import apply_showcase, match_showcase


def test_gatsby_pins_lana():
    entry = match_showcase("The Great Gatsby", "book")
    assert entry is not None
    result = apply_showcase(
        {"found": False, "query": "The Great Gatsby", "recommendations": []},
        query="The Great Gatsby",
        input_type="book",
        want="all",
    )
    assert result["found"]
    assert result["matched_title"] == "The Great Gatsby"
    creators = [r["creator"] for r in result["recommendations"]]
    assert any("Lana Del Rey" in c for c in creators)
    assert any(r["type"] == "book" for r in result["recommendations"])


def test_midnights_pins_normal_people():
    result = apply_showcase(
        {"found": False, "query": "Midnight by Taylor Swift", "recommendations": []},
        query="Midnight by Taylor Swift",
        input_type="song",
        want="all",
    )
    assert result["matched_title"] == "Midnights"
    titles = [r["title"] for r in result["recommendations"]]
    assert "Normal People" in titles


def test_dune_not_messiah_pins_zimmer():
    result = apply_showcase(
        {
            "found": True,
            "matched_title": "Dune Messiah (Dune Chronicles #2)",
            "matched_creator": "Frank Herbert",
            "matched_type": "book",
            "recommendations": [{"title": "Other", "creator": "X", "type": "song"}],
        },
        query="Dune by Frank Herbert",
        input_type="book",
        want="song",
    )
    assert result["matched_title"] == "Dune"
    assert result["recommendations"][0]["creator"] == "Hans Zimmer"
    assert result["recommendations"][0]["title"] == "Time"
