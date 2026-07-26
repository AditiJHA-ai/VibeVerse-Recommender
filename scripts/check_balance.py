import json
import urllib.request


def rec(query, input_type="book", want="all"):
    body = json.dumps(
        {
            "query": query,
            "input_type": input_type,
            "want": want,
            "top_n": 8,
            "allow_live": True,
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8001/api/recommend",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)
    books = [x for x in data.get("recommendations", []) if x.get("type") == "book"]
    songs = [x for x in data.get("recommendations", []) if x.get("type") == "song"]
    print(
        f"== {query} ({input_type}, want={want}) "
        f"matched={data.get('matched_title')} books={len(books)} songs={len(songs)}"
    )
    if want == "all":
        print("   books:", ", ".join(x["title"][:40] for x in books[:3]) or "(none)")
        print("   songs:", ", ".join(x["title"][:40] for x in songs[:3]) or "(none)")
        print("   FAIL" if len(books) < 1 or len(songs) < 1 else "   OK")
    elif want == "book":
        print("   OK" if books and not songs else f"   FAIL books={len(books)} songs={len(songs)}")
    elif want == "song":
        print("   OK" if songs and not books else f"   FAIL books={len(books)} songs={len(songs)}")
    return data


CASES = [
    ("The Great Gatsby", "book", "all"),
    ("Midnight by Taylor Swift", "song", "all"),
    ("Dune by Frank Herbert", "book", "all"),
    ("Fourth Wing", "book", "all"),
    ("Cruel Summer", "song", "all"),
    ("The Odyssey", "book", "all"),
    ("The Great Gatsby", "book", "song"),
    ("The Great Gatsby", "book", "book"),
]

for q, t, w in CASES:
    try:
        rec(q, t, w)
    except Exception as e:
        print("==", q, "ERROR", e)
