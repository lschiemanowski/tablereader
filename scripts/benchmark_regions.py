"""Read table_benchmark's v1 release as document_prep inputs (shared by the recrop and check scripts).

Read-only: nothing is written into the benchmark repository.
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


def load_pages(benchmark: Path) -> list[Page]:
    release = benchmark / "data/full/release-v1"
    documents = {d["document_id"]: d for d in map(json.loads, (release / "documents.jsonl").open())}
    rotation: dict[tuple[str, int], int] = {}
    for line in (release / "cell-references.jsonl").open():
        ref = json.loads(line)
        for p in ref["parts"]:
            rotation[(ref["crop_key"], p["part"])] = p.get("rotation_cw", 0) or 0
    detections: dict[tuple[str, int], list[Box]] = defaultdict(list)
    for path in sorted((benchmark / "data/full/layout").glob("*.json")):
        layout = json.loads(path.read_text())
        for t in layout["tables"]:
            detections[(layout["document_id"], t["page"])].append(_box(t["bbox"]))

    by_page: dict[tuple[str, int], list[Part]] = defaultdict(list)
    for line in (release / "examples.jsonl").open():
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
