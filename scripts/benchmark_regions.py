"""Read table_benchmark's v1 release, or its scan expansion, as document_prep inputs (shared by the recrop and
check scripts).

The scan expansion (`data/full/scan-expansion`, 2026-11-13) adds 1,465 scanned validation and test tables from new
series; with v1's validation and test it forms the scan-balanced evaluation export. Read-only: nothing is written
into the benchmark repository.
"""

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_prep.models import Box, TableRegion


@dataclass(frozen=True)
class Part:
    """One crop part of a benchmark example (one page of it)."""

    example: dict[str, Any]
    part: dict[str, Any]
    region: TableRegion


@dataclass(frozen=True)
class Page:
    document_id: str
    pdf: Path
    sha256: str
    page: int
    parts: tuple[Part, ...]
    other_boxes: tuple[Box, ...]  # detections on the page that are not benchmark parts


def _box(values: list[float]) -> Box:
    return Box(left=values[0], top=values[1], right=values[2], bottom=values[3])


def _iou(a: Box, b: Box) -> float:
    w = min(a.right, b.right) - max(a.left, b.left)
    h = min(a.bottom, b.bottom) - max(a.top, b.top)
    inter = max(w, 0) * max(h, 0)
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
    return inter / union if union else 0.0


SOURCES = ("v1", "scan-expansion")


def _detections(directory: Path) -> dict[tuple[str, int], list[Box]]:
    detections: dict[tuple[str, int], list[Box]] = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        layout = json.loads(path.read_text())
        for t in layout["tables"]:
            detections[(layout["document_id"], t["page"])].append(_box(t["bbox"]))
    return detections


def load_pages(benchmark: Path, source: str = "v1") -> list[Page]:
    """The pages holding crop parts of `source`: "v1" (the frozen release) or "scan-expansion"."""
    if source == "v1":
        directory = benchmark / "data/full/release-v1"
        examples, references, layout = (
            directory / "examples.jsonl",
            directory / "cell-references.jsonl",
            benchmark / "data/full/layout",
        )
    elif source == "scan-expansion":
        directory = benchmark / "data/full/scan-expansion"
        examples, references, layout = (
            directory / "selected.jsonl",
            directory / "reference-grids.jsonl",
            directory / "layout",
        )
    else:
        raise ValueError(f"unknown benchmark source {source}; sources are {', '.join(SOURCES)}")
    documents = {d["document_id"]: d for d in map(json.loads, (directory / "documents.jsonl").open())}
    rotation: dict[tuple[str, int], int] = {}
    for line in references.open():
        ref = json.loads(line)
        for p in ref["parts"]:
            rotation[(ref["crop_key"], p["part"])] = p.get("rotation_cw", 0) or 0
    detections = _detections(layout)

    by_page: dict[tuple[str, int], list[Part]] = defaultdict(list)
    for line in examples.open():
        example = json.loads(line)
        for p in example["parts"]:
            region = TableRegion(
                region_id=f"{example['crop_key']}-{p['part']}",
                page=p["pdf_page"],
                box=_box(p["bbox"]),
                rotation_cw=rotation.get((example["crop_key"], p["part"]), 0),
            )
            by_page[(example["document_id"], p["pdf_page"])].append(Part(example, p, region))

    pages: list[Page] = []
    for (document_id, page), parts in sorted(by_page.items()):
        doc = documents[document_id]
        boxes = [p.region.box for p in parts]
        others = tuple(d for d in detections.get((document_id, page), []) if all(_iou(d, b) < 0.99 for b in boxes))
        pages.append(Page(document_id, benchmark / doc["path"], doc["sha256"], page, tuple(parts), others))
    return pages


def iter_parts(pages: list[Page]) -> Iterator[tuple[Page, Part]]:
    for page in pages:
        for part in page.parts:
            yield page, part
