"""Heron layout detection (only with the layout extra installed and the pinned model in the local cache)."""

import os

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
pytest.importorskip("transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from document_prep.heron import HeronLayoutDetector

LABELS = {
    "caption", "footnote", "formula", "list_item", "page_footer", "page_header", "picture", "section_header",
    "table", "text", "title", "document_index", "code", "checkbox_selected", "checkbox_unselected", "form",
    "key_value_region",
}  # fmt: skip


@pytest.fixture(scope="module")
def detector() -> HeronLayoutDetector:
    try:
        return HeronLayoutDetector(device="cpu", threshold=0.3)
    except OSError as error:  # model not in the local cache
        pytest.skip(f"Heron model unavailable: {error}")


def test_detects_regions_in_page_pixels(detector: HeronLayoutDetector) -> None:
    page = Image.new("RGB", (612, 792), "white")
    draw = ImageDraw.Draw(page)
    draw.text((72, 60), "Table 1. Receipts and expenditures", fill="black")
    for row in range(12):
        y = 100 + row * 18
        draw.line((72, y, 540, y), fill="black")
        for x in (80, 260, 380, 480):
            draw.text((x, y + 4), f"{row * 17 + x:,}", fill="black")
    (regions,) = detector.detect([page])
    assert regions
    assert {r.label for r in regions} <= LABELS
    assert all(0 <= r.box.left <= r.box.right <= 612 and 0 <= r.box.top <= r.box.bottom <= 792 for r in regions)
    assert all(r.score >= 0.3 for r in regions)
