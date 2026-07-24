"""Crop illustration regions from Gamma landing screenshots."""

from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "assets" / "gamma"

CROPS = {
    "image-60081faf-3983-4285-8749-9d187b669f01.png": (530, 20, 1000, 470, "hero-books.png"),
    "image-af16d971-50e5-4516-a07a-d4b9bbf2b1af.png": (540, 40, 1000, 480, "two-questions.png"),
    "image-ab5ad764-74fb-4ebe-8085-9d1294361161.png": (280, 220, 760, 540, "vibe-wheel.png"),
    "image-af5c7dda-6172-4306-b1dd-a5b1f7ec40a1.png": (500, 20, 1000, 430, "cross-domain.png"),
    "image-9d418a46-9831-4551-adc9-ba441bcf9f4e.png": (520, 200, 1000, 545, "ready-explore.png"),
}


def main() -> None:
    for src, (left, upper, right, lower, out) in CROPS.items():
        im = Image.open(BASE / src).convert("RGB")
        w, h = im.size
        box = (max(0, left), max(0, upper), min(w, right), min(h, lower))
        crop = im.crop(box)
        crop.save(BASE / out, quality=95)
        print(out, crop.size)


if __name__ == "__main__":
    main()
