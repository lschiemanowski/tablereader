"""Recrop table_benchmark v1 with document_prep: every part rendered at 300 DPI and extended over its title
and footnotes, sideways tables turned upright.

Reads the benchmark (read-only) and writes into --out: `images/<region_id>.png` and `manifest.jsonl`, one line
per part in benchmark order, with the boxes, the settings and the renderer. Rerunning with the same versions
gives byte-identical images.

    uv run python scripts/recrop_benchmark.py ~/workspace/table_benchmark --workers 48
"""

import argparse
import dataclasses
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_regions import Page, load_pages

from document_prep.models import Box, DocumentSource, LayoutRegion, PreparationSettings
from document_prep.prepare import prepare_page


def _box(box: Box, digits: int = 2) -> list[float]:
    return [round(v, digits) for v in (box.left, box.top, box.right, box.bottom)]


def recrop_page(args: tuple[Page, Path, PreparationSettings, list[LayoutRegion]]) -> list[dict[str, Any]]:
    page, out, settings, layout = args
    source = DocumentSource(path=page.pdf, sha256=page.sha256)
    regions = [p.region for p in page.parts]
    prepared = prepare_page(source, page.page, regions, settings, other_boxes=page.other_boxes, layout=layout)
    rows: list[dict[str, Any]] = []
    for part, crop in zip(page.parts, prepared, strict=True):
        buffer = io.BytesIO()
        crop.image.save(buffer, "PNG")
        data = buffer.getvalue()
        name = f"images/{crop.region.region_id}.png"
        (out / name).write_bytes(data)
        example = part.example
        rows.append(
            {
                "table_id": example["table_id"],
                "crop_key": example["crop_key"],
                "part": part.part["part"],
                "split": example["split"],
                "label": example["label"],
                "source_id": example["source_id"],
                "medium": example["medium"],
                "document_id": page.document_id,
                "page": page.page,
                "rotation_cw": crop.region.rotation_cw,
                "detected_box": _box(crop.region.box),
                "crop_box": _box(crop.crop_box),
                "page_size": [round(v, 2) for v in crop.page_size],
                "detected_px": _box(crop.detected_px, 1),
                "crop_px": _box(crop.crop_px, 1),
                "page_pixels": list(crop.page_pixels),
                "image": name,
                "image_size": list(crop.image.size),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def load_layout(directory: Path, settings: PreparationSettings) -> dict[tuple[str, int], list[LayoutRegion]]:
    """Layout regions per (document, page), in points, from benchmark_layout.py; checked against the settings."""
    found: dict[tuple[str, int], list[LayoutRegion]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload["model"] != settings.layout_model or payload["dpi"] != settings.layout_dpi:
            raise SystemExit(f"{path} was made with other layout settings: {payload['model']} at {payload['dpi']} DPI")
        for page, regions in payload["pages"].items():
            found[(path.stem, int(page))] = [
                LayoutRegion(
                    label=r["label"],
                    score=r["score"],
                    box=Box(left=r["box"][0], top=r["box"][1], right=r["box"][2], bottom=r["box"][3]),
                )
                for r in regions
            ]
    if not found:
        raise SystemExit(f"no layout in {directory}; run scripts/benchmark_layout.py first")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/recrops"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ids", type=Path, help="only parts whose region_id or table_id is listed in this file")
    parser.add_argument("--layout", type=Path, default=Path("data/layout"), help="output of benchmark_layout.py")
    args = parser.parse_args()
    settings = PreparationSettings()
    pages = load_pages(args.benchmark)
    if args.ids:
        wanted = set(args.ids.read_text().split())

        def keep(x: Any) -> bool:
            return x.region.region_id in wanted or x.example["table_id"] in wanted

        # Parts left out still bound the crops of the parts kept, as in a full run.
        pages = [
            dataclasses.replace(
                p,
                parts=tuple(x for x in p.parts if keep(x)),
                other_boxes=p.other_boxes + tuple(x.region.box for x in p.parts if not keep(x)),
            )
            for p in pages
        ]
        pages = [p for p in pages if p.parts]
    for document_id, path, sha256 in sorted({(p.document_id, p.pdf, p.sha256) for p in pages}):
        if DocumentSource.from_path(path).sha256 != sha256:
            raise SystemExit(f"{path} differs from the PDF the benchmark was built from ({document_id})")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    layout = load_layout(args.layout, settings)
    jobs = [(p, args.out, settings, layout.get((p.document_id, p.page), [])) for p in pages]
    with ProcessPoolExecutor(args.workers) as pool:
        for page_rows in pool.map(recrop_page, jobs, chunksize=4):
            rows.extend(page_rows)
    order = {
        (json.loads(line)["crop_key"]): i
        for i, line in enumerate((args.benchmark / "data/full/release-v1/examples.jsonl").open())
    }
    rows.sort(key=lambda r: (order[r["crop_key"]], r["part"]))
    from document_prep.prepare import RENDERER

    header = {"settings": dataclasses.asdict(settings), "renderer": RENDERER}
    (args.out / "settings.json").write_text(json.dumps(header, indent=2) + "\n")
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} parts on {len(pages)} pages -> {args.out}")


if __name__ == "__main__":
    main()
