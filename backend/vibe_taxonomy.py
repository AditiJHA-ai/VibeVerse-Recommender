"""
Shared vibe taxonomy bridging books and songs.

Landing-page primaries (Intimate / Electric / Dreamy / Epic) plus finer vibes
used for matching. Tags from either domain map into the same vector space.
"""

from __future__ import annotations

from typing import Iterable

# Order is fixed — vectors are aligned to this list.
VIBES: list[str] = [
    # Primary (landing page)
    "intimate",
    "electric",
    "dreamy",
    "epic",
    # Fine-grained
    "melancholy",
    "euphoric",
    "cozy",
    "intense",
    "nostalgic",
    "romantic",
    "rebellious",
    "chill",
    "dark",
    "whimsical",
    "adventurous",
    "contemplative",
]

VIBE_INDEX = {v: i for i, v in enumerate(VIBES)}
N_VIBES = len(VIBES)

# Goodreads / shelf noise — never use for matching
SHELF_NOISE = frozenset(
    {
        "to-read",
        "currently-reading",
        "favorites",
        "favourites",
        "owned",
        "books-i-own",
        "owned-books",
        "kindle",
        "ebook",
        "ebooks",
        "to-buy",
        "library",
        "default",
        "audiobook",
        "audiobooks",
        "audio",
        "book-club",
        "tbr",
        "dnf",
        "series",
        "sries",
        "novels",
        "adult",
        "read",
        "unread",
        "abandoned",
        "maybe",
        "wish-list",
        "wishlist",
        "my-books",
        "my-library",
        "physical",
        "hardcover",
        "paperback",
        "arc",
        "netgalley",
        "owlcrate",
        "bookish",
        "bookshelf",
        "home-library",
        "re-read",
        "reread",
        "reviewed",
        "borrowed",
        "from-library",
    }
)

# tag (normalized, hyphen/underscore collapsed) → {vibe: weight}
# Weights are soft scores in [0, 1]; multiple vibes per tag are allowed.
TAG_TO_VIBES: dict[str, dict[str, float]] = {
    # --- Song mood_* tags (AcousticBrainz-style) ---
    "mood_sad": {"melancholy": 1.0, "intimate": 0.6, "dark": 0.3},
    "mood_energetic": {"electric": 1.0, "euphoric": 0.5, "intense": 0.4},
    "mood_happy": {"euphoric": 1.0, "electric": 0.4, "whimsical": 0.3},
    "mood_positive": {"euphoric": 0.8, "chill": 0.3},
    "mood_negative": {"melancholy": 0.8, "dark": 0.5, "intense": 0.3},
    "mood_calm": {"chill": 1.0, "cozy": 0.7, "intimate": 0.5, "contemplative": 0.4},
    "mood_acoustic": {"intimate": 0.9, "cozy": 0.6, "contemplative": 0.4},
    "mood_danceable": {"electric": 1.0, "euphoric": 0.6},
    # --- Book mood* tags ---
    "moodromantic": {"romantic": 1.0, "intimate": 0.8},
    "mood_romantic": {"romantic": 1.0, "intimate": 0.8},
    "moodsuspense": {"intense": 1.0, "electric": 0.6, "dark": 0.3},
    "mood_suspense": {"intense": 1.0, "electric": 0.6},
    "moodadventurous": {"adventurous": 1.0, "epic": 0.7, "electric": 0.4},
    "mood_adventurous": {"adventurous": 1.0, "epic": 0.7},
    "mooddark": {"dark": 1.0, "intense": 0.5, "melancholy": 0.3},
    "moodfunny": {"whimsical": 0.9, "euphoric": 0.4},
    "moodmysterious": {"dark": 0.6, "dreamy": 0.5, "intense": 0.4},
    "moodhopeful": {"euphoric": 0.6, "dreamy": 0.4, "romantic": 0.3},
    "moodemotional": {"intimate": 0.8, "melancholy": 0.5, "romantic": 0.4},
    # --- Genres / content (books) ---
    "fantasy": {"dreamy": 0.8, "epic": 0.7, "adventurous": 0.5},
    "high-fantasy": {"epic": 1.0, "dreamy": 0.7, "adventurous": 0.6},
    "ya-fantasy": {"dreamy": 0.7, "epic": 0.5, "romantic": 0.4, "electric": 0.3},
    "urban-fantasy": {"electric": 0.6, "dark": 0.5, "dreamy": 0.4},
    "young-adult": {"electric": 0.4, "romantic": 0.3, "intense": 0.3},
    "ya": {"electric": 0.4, "romantic": 0.3},
    "romance": {"romantic": 1.0, "intimate": 0.7},
    "chick-lit": {"romantic": 0.7, "whimsical": 0.5, "cozy": 0.3},
    "contemporary": {"intimate": 0.5, "contemplative": 0.3},
    "contemporary-romance": {"romantic": 0.9, "intimate": 0.7},
    "historical-fiction": {"nostalgic": 0.7, "epic": 0.4, "contemplative": 0.4},
    "historical": {"nostalgic": 0.6, "epic": 0.3},
    "classics": {"nostalgic": 0.7, "contemplative": 0.8, "epic": 0.3},
    "classic": {"nostalgic": 0.7, "contemplative": 0.8},
    "science-fiction": {"epic": 0.8, "dreamy": 0.6, "adventurous": 0.5},
    "sci-fi": {"epic": 0.8, "dreamy": 0.6, "adventurous": 0.5},
    "horror": {"dark": 1.0, "intense": 0.8},
    "thriller": {"intense": 1.0, "electric": 0.7, "dark": 0.4},
    "mystery": {"intense": 0.6, "dark": 0.4, "contemplative": 0.3},
    "suspense": {"intense": 0.9, "electric": 0.5},
    "paranormal": {"dreamy": 0.6, "dark": 0.5, "romantic": 0.3},
    "magic": {"dreamy": 0.8, "whimsical": 0.4, "epic": 0.3},
    "humor": {"whimsical": 0.9, "euphoric": 0.4},
    "comedy": {"whimsical": 0.9, "euphoric": 0.5},
    "biography": {"contemplative": 0.8, "intimate": 0.5, "nostalgic": 0.3},
    "memoir": {"intimate": 0.9, "contemplative": 0.7, "nostalgic": 0.5},
    "non-fiction": {"contemplative": 0.6},
    "nonfiction": {"contemplative": 0.6},
    "poetry": {"intimate": 0.8, "melancholy": 0.4, "dreamy": 0.5},
    "literary-fiction": {"contemplative": 0.8, "intimate": 0.6, "melancholy": 0.3},
    "dystopia": {"dark": 0.8, "intense": 0.7, "epic": 0.4},
    "dystopian": {"dark": 0.8, "intense": 0.7, "epic": 0.4},
    "adventure": {"adventurous": 1.0, "epic": 0.7, "electric": 0.4},
    "war": {"epic": 0.8, "intense": 0.7, "dark": 0.5},
    "mythology": {"epic": 1.0, "dreamy": 0.6, "adventurous": 0.5},
    "myths": {"epic": 0.9, "dreamy": 0.5},
    "retelling": {"dreamy": 0.5, "epic": 0.4, "nostalgic": 0.4},
    "fairy-tales": {"whimsical": 0.7, "dreamy": 0.8},
    "childrens": {"whimsical": 0.8, "cozy": 0.4},
    "children": {"whimsical": 0.8, "cozy": 0.4},
    "middle-grade": {"whimsical": 0.6, "adventurous": 0.5},
    "fiction": {"contemplative": 0.15},  # weak — almost every book
    # --- Song genres / styles ---
    "pop": {"euphoric": 0.5, "electric": 0.4, "romantic": 0.3},
    "indie-pop": {"dreamy": 0.5, "intimate": 0.4, "nostalgic": 0.3},
    "indie": {"intimate": 0.4, "dreamy": 0.4, "melancholy": 0.3},
    "folk": {"intimate": 0.7, "cozy": 0.5, "contemplative": 0.5, "nostalgic": 0.4},
    "singer-songwriter": {"intimate": 0.9, "contemplative": 0.6, "melancholy": 0.3},
    "acoustic": {"intimate": 0.7, "cozy": 0.5, "chill": 0.4},
    "soul": {"intimate": 0.6, "romantic": 0.5, "nostalgic": 0.4},
    "blues": {"melancholy": 0.8, "nostalgic": 0.6, "intimate": 0.4},
    "jazz": {"chill": 0.5, "nostalgic": 0.6, "intimate": 0.4, "dreamy": 0.3},
    "classical": {"epic": 0.5, "contemplative": 0.7, "dreamy": 0.4},
    "orchestral": {"epic": 0.9, "dreamy": 0.5},
    "soundtrack": {"epic": 0.6, "dreamy": 0.5, "intense": 0.3},
    "metal": {"intense": 0.9, "dark": 0.7, "rebellious": 0.8, "electric": 0.5},
    "black-metal": {"dark": 1.0, "intense": 0.9, "rebellious": 0.7},
    "grindcore": {"intense": 1.0, "dark": 0.8, "rebellious": 0.9, "electric": 0.5},
    "industrial": {"dark": 0.7, "electric": 0.6, "intense": 0.6},
    "punk": {"rebellious": 1.0, "electric": 0.8, "intense": 0.5},
    "hip-hop": {"electric": 0.6, "rebellious": 0.4, "intense": 0.3},
    "rap": {"electric": 0.5, "rebellious": 0.4},
    "edm": {"electric": 1.0, "euphoric": 0.8},
    "dubstep": {"electric": 0.9, "intense": 0.6, "dark": 0.3},
    "dance": {"electric": 0.9, "euphoric": 0.7},
    "club": {"electric": 0.8, "euphoric": 0.6},
    "house": {"electric": 0.7, "euphoric": 0.5, "chill": 0.2},
    "ambient": {"dreamy": 0.9, "chill": 0.8, "contemplative": 0.5},
    "dream-pop": {"dreamy": 1.0, "intimate": 0.4},
    "shoegaze": {"dreamy": 0.9, "melancholy": 0.5},
    "lo-fi": {"chill": 0.9, "cozy": 0.7, "nostalgic": 0.4},
    "lofi": {"chill": 0.9, "cozy": 0.7},
    "rnb": {"romantic": 0.6, "intimate": 0.5, "chill": 0.3},
    "r&b": {"romantic": 0.6, "intimate": 0.5},
    "disney": {"whimsical": 1.0, "euphoric": 0.4, "dreamy": 0.4},
    "kids": {"whimsical": 0.9, "euphoric": 0.3},
    "musical": {"epic": 0.7, "euphoric": 0.5, "romantic": 0.3, "intense": 0.3},
    "broadway": {"epic": 0.8, "euphoric": 0.5, "romantic": 0.3},
    "theatre": {"epic": 0.7, "intense": 0.4},
    "theater": {"epic": 0.7, "intense": 0.4},
    "opera": {"epic": 0.9, "intense": 0.5, "romantic": 0.3},
    "world-music": {"adventurous": 0.4, "dreamy": 0.3},
    "gospel": {"euphoric": 0.6, "intimate": 0.4, "epic": 0.3},
    "worship": {"intimate": 0.5, "euphoric": 0.4, "contemplative": 0.4},
    # --- Google Books / Last.fm free-text style tags ---
    "romance": {"romantic": 1.0, "intimate": 0.7},
    "love": {"romantic": 0.8, "intimate": 0.6},
    "sad": {"melancholy": 0.9, "intimate": 0.5},
    "melancholy": {"melancholy": 1.0, "intimate": 0.5},
    "dark": {"dark": 1.0, "intense": 0.4},
    "epic": {"epic": 1.0, "adventurous": 0.5},
    "adventure": {"adventurous": 1.0, "epic": 0.6},
    "atmospheric": {"dreamy": 0.7, "chill": 0.4},
    "ethereal": {"dreamy": 1.0, "chill": 0.3},
    "uplifting": {"euphoric": 0.9, "electric": 0.3},
    "aggressive": {"intense": 0.9, "rebellious": 0.7, "electric": 0.5},
    "peaceful": {"chill": 0.9, "cozy": 0.6},
    "cozy": {"cozy": 1.0, "intimate": 0.5},
    "nostalgia": {"nostalgic": 1.0},
    "nostalgic": {"nostalgic": 1.0},
    "rebellious": {"rebellious": 1.0, "electric": 0.4},
    "intense": {"intense": 1.0, "electric": 0.4},
    "chill": {"chill": 1.0, "cozy": 0.4},
    "dreamy": {"dreamy": 1.0},
    "intimate": {"intimate": 1.0},
    "electric": {"electric": 1.0},
    "mythology": {"epic": 1.0, "dreamy": 0.5},
    "greek-mythology": {"epic": 1.0, "dreamy": 0.6, "adventurous": 0.5},
    "homeric": {"epic": 1.0, "adventurous": 0.7},
    "dragons": {"epic": 0.8, "dreamy": 0.7, "adventurous": 0.6},
    "romantasy": {"romantic": 0.9, "dreamy": 0.7, "epic": 0.5, "electric": 0.4},
    "enemies-to-lovers": {"romantic": 0.9, "intense": 0.5, "electric": 0.4},
}

# Keyword hints for free-text (book descriptions, Last.fm tags as phrases)
TEXT_HINTS: list[tuple[str, dict[str, float]]] = [
    ("odyssey", {"epic": 1.0, "adventurous": 0.9, "nostalgic": 0.4}),
    ("homer", {"epic": 0.9, "adventurous": 0.7}),
    ("mythology", {"epic": 0.9, "dreamy": 0.5}),
    ("dragon", {"epic": 0.7, "dreamy": 0.6, "adventurous": 0.5}),
    ("war college", {"epic": 0.5, "intense": 0.6, "electric": 0.4}),
    ("fourth wing", {"romantic": 0.9, "epic": 0.7, "electric": 0.6, "dreamy": 0.5, "intense": 0.4}),
    ("iron flame", {"romantic": 0.8, "epic": 0.7, "intense": 0.5, "electric": 0.5}),
    ("romantasy", {"romantic": 0.9, "dreamy": 0.7, "epic": 0.5}),
    ("dragon rider", {"epic": 0.8, "adventurous": 0.7, "dreamy": 0.5, "romantic": 0.3}),
    ("enemies to lovers", {"romantic": 0.9, "intense": 0.5}),
    ("alex warren", {"romantic": 0.8, "intimate": 0.7, "melancholy": 0.4}),
    ("epic the musical", {"epic": 1.0, "adventurous": 0.8, "intense": 0.5, "nostalgic": 0.3}),
    ("jorge rivera", {"epic": 0.9, "adventurous": 0.7}),
    ("musical", {"epic": 0.7, "euphoric": 0.4, "romantic": 0.3}),
    ("broadway", {"epic": 0.8, "euphoric": 0.4}),
    ("soundtrack of", {"epic": 0.5, "dreamy": 0.3}),
    ("melanchol", {"melancholy": 0.9, "intimate": 0.5}),
    ("bittersweet", {"melancholy": 0.7, "nostalgic": 0.6, "romantic": 0.4}),
    ("heartbreak", {"melancholy": 0.8, "romantic": 0.6, "intimate": 0.5}),
    ("uplifting", {"euphoric": 0.8}),
    ("dystopian", {"dark": 0.8, "intense": 0.6}),
    ("space opera", {"epic": 0.9, "adventurous": 0.7, "dreamy": 0.4}),
    ("cozy mystery", {"cozy": 0.8, "whimsical": 0.4}),
    ("coming of age", {"intimate": 0.6, "nostalgic": 0.5, "contemplative": 0.4}),
]


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    t = t.replace(" ", "-").replace("_", "-")
    # Keep mood_* song tags distinguishable: restore underscore form for known moods
    mood_underscored = {
        "mood-sad": "mood_sad",
        "mood-energetic": "mood_energetic",
        "mood-happy": "mood_happy",
        "mood-positive": "mood_positive",
        "mood-negative": "mood_negative",
        "mood-calm": "mood_calm",
        "mood-acoustic": "mood_acoustic",
        "mood-danceable": "mood_danceable",
        "mood-romantic": "mood_romantic",
        "mood-suspense": "mood_suspense",
        "mood-adventurous": "mood_adventurous",
    }
    if t in mood_underscored:
        return mood_underscored[t]
    # book mood tags are often glued: moodromantic
    return t.replace("-", "") if t.startswith("mood") and "-" in t else t


def is_noise_tag(tag: str) -> bool:
    t = tag.strip().lower()
    if t in SHELF_NOISE:
        return True
    # year / read-in / release housekeeping
    if t.startswith("read-in-") or t.startswith("read-"):
        return True
    if t.endswith("-releases") or t.endswith("-release"):
        return True
    if t.isdigit() and len(t) == 4:
        return True
    if t.startswith("owned"):
        return True
    return False


def clean_tags(raw_tags: Iterable[str] | str) -> list[str]:
    if isinstance(raw_tags, str):
        parts = raw_tags.replace(",", " ").split()
    else:
        parts = list(raw_tags)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p or is_noise_tag(p):
            continue
        # keep multi-word tags intact when they used hyphens originally
        norm = p.strip().lower()
        if is_noise_tag(norm):
            continue
        key = normalize_tag(norm)
        if key in seen:
            continue
        seen.add(key)
        out.append(norm if "-" in norm or "_" in norm else key)
    return out


def tags_to_vibe_vector(tags: Iterable[str], text: str = "") -> list[float]:
    vec = [0.0] * N_VIBES
    for tag in tags:
        key = normalize_tag(tag)
        # try exact, then hyphenated, then de-hyphenated
        mapping = (
            TAG_TO_VIBES.get(key)
            or TAG_TO_VIBES.get(tag.strip().lower())
            or TAG_TO_VIBES.get(key.replace("_", "-"))
            or TAG_TO_VIBES.get(key.replace("-", "_"))
        )
        if not mapping and key.startswith("mood") and not key.startswith("mood_"):
            # moodromantic style
            mapping = TAG_TO_VIBES.get(key)
        if mapping:
            for vibe, w in mapping.items():
                if vibe in VIBE_INDEX:
                    idx = VIBE_INDEX[vibe]
                    vec[idx] = min(1.0, vec[idx] + w)

    blob = (text or "").lower()
    if blob:
        for needle, mapping in TEXT_HINTS:
            if needle in blob:
                for vibe, w in mapping.items():
                    if vibe in VIBE_INDEX:
                        idx = VIBE_INDEX[vibe]
                        vec[idx] = min(1.0, vec[idx] + w)

    return vec


def primary_vibe(vector: list[float]) -> str:
    if not any(vector):
        return "dreamy"
    primaries = ["intimate", "electric", "dreamy", "epic"]
    best = max(primaries, key=lambda v: vector[VIBE_INDEX[v]])
    if vector[VIBE_INDEX[best]] <= 0:
        # fall back to any top vibe, map to nearest primary
        top = VIBES[max(range(N_VIBES), key=lambda i: vector[i])]
        fallback = {
            "melancholy": "intimate",
            "romantic": "intimate",
            "cozy": "intimate",
            "contemplative": "intimate",
            "euphoric": "electric",
            "intense": "electric",
            "rebellious": "electric",
            "chill": "dreamy",
            "whimsical": "dreamy",
            "nostalgic": "dreamy",
            "dark": "epic",
            "adventurous": "epic",
        }
        return fallback.get(top, "dreamy")
    return best


def top_vibes(vector: list[float], k: int = 3) -> list[str]:
    ranked = sorted(range(N_VIBES), key=lambda i: vector[i], reverse=True)
    return [VIBES[i] for i in ranked[:k] if vector[i] > 0]
