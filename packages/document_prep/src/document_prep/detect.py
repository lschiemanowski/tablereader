"""Table detection: Docling Heron's `table` regions become `TableRegion`s.

`table_regions` is pure: it keeps `table` regions at or above a score threshold, drops a detection that lies
mostly inside a higher-scoring one, and orders the rest in reading order (top to bottom, then left to right).
Every region is taken as upright; sideways tables are not detected as such yet. `detect_layout` renders pages at
the settings' layout resolution and runs a detector on them, returning boxes in points.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol

from PIL import Image

from document_prep.models import POINTS_PER_INCH, Box, DocumentSource, LayoutRegion, PreparationSettings, TableRegion
from document_prep.prepare import render_page

TABLE_LABEL = "table"
TABLE_THRESHOLD = 0.5
CONTAINED = 0.8  # a detection with this share of its area inside a better one is a duplicate


class LayoutDetector(Protocol):
    """Layout regions of page images, boxes in each image's pixels (`HeronLayoutDetector` implements it)."""

    def detect(self, images: Sequence[Image.Image]) -> list[list[LayoutRegion]]: ...


def _area(box: Box) -> float:
    return max(0.0, box.right - box.left) * max(0.0, box.bottom - box.top)


def _inside_share(inner: Box, outer: Box) -> float:
    left, top = max(inner.left, outer.left), max(inner.top, outer.top)
    right, bottom = min(inner.right, outer.right), min(inner.bottom, outer.bottom)
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    return overlap / _area(inner) if _area(inner) else 0.0


def table_regions(
    layout: Iterable[LayoutRegion], page: int, *, threshold: float = TABLE_THRESHOLD
) -> list[TableRegion]:
    """The page's tables in reading order, with ids `p<page>-t<n>`."""
    found = sorted(
        (r for r in layout if r.label == TABLE_LABEL and r.score >= threshold), key=lambda r: r.score, reverse=True
    )
    kept: list[LayoutRegion] = []
    for region in found:
        if all(_inside_share(region.box, k.box) < CONTAINED for k in kept):
            kept.append(region)
    kept.sort(key=lambda r: (round(r.box.top), r.box.left))
    return [TableRegion(region_id=f"p{page:04d}-t{i}", page=page, box=r.box) for i, r in enumerate(kept)]


def detect_layout(
    detector: LayoutDetector, source: DocumentSource, pages: Sequence[int], settings: PreparationSettings
) -> dict[int, list[LayoutRegion]]:
    """Layout regions of `pages`, boxes in points, rendered at `settings.layout_dpi`."""
    images = [render_page(source, page, settings.layout_dpi)[0] for page in pages]
    scale = POINTS_PER_INCH / settings.layout_dpi
    detected = detector.detect(images)
    return {
        page: [LayoutRegion(label=r.label, score=r.score, box=r.box.scaled(scale)) for r in regions]
        for page, regions in zip(pages, detected, strict=True)
    }
