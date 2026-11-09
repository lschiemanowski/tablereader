"""Heron table regions to TableRegions: threshold, duplicates, reading order."""

from document_prep.detect import table_regions
from document_prep.models import Box, LayoutRegion


def region(label: str, score: float, left: float, top: float, right: float, bottom: float) -> LayoutRegion:
    return LayoutRegion(label=label, score=score, box=Box(left=left, top=top, right=right, bottom=bottom))


def test_tables_above_threshold_in_reading_order() -> None:
    layout = [
        region("table", 0.9, 300, 100, 500, 300),  # right column
        region("text", 0.99, 0, 0, 500, 50),
        region("table", 0.8, 0, 100, 280, 300),  # left column, same row
        region("table", 0.95, 0, 400, 500, 600),
        region("table", 0.3, 0, 700, 500, 800),  # below threshold
    ]
    tables = table_regions(layout, page=3)
    assert [t.region_id for t in tables] == ["p0003-t0", "p0003-t1", "p0003-t2"]
    assert [(t.box.left, t.box.top) for t in tables] == [(0, 100), (300, 100), (0, 400)]
    assert all(t.page == 3 and t.rotation_cw == 0 for t in tables)


def test_a_detection_inside_a_better_one_is_dropped() -> None:
    layout = [region("table", 0.95, 0, 0, 500, 500), region("table", 0.7, 10, 10, 490, 250)]
    (table,) = table_regions(layout, page=1)
    assert table.box == Box(left=0, top=0, right=500, bottom=500)
