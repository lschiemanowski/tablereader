"""Run Docling Heron layout detection on every benchmark page that holds a crop part.

Pages are rendered at the settings' layout resolution (72 DPI) in worker processes and detected in batches on
the GPU. Writes `<out>/<document_id>.json`: model, threshold, and per page the regions (label, score, box in
points). Needs the layout extra: `uv sync --group layout`.

    uv run python scripts/benchmark_layout.py path/to/table_benchmark --workers 32
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_regions import load_pages
from PIL import Image

from document_prep.models import DocumentSource, PreparationSettings
from document_prep.prepare import render_page


def render(args: tuple[Path, str, int, int]) -> tuple[Image.Image, tuple[float, float]]:
    path, sha256, page, dpi = args
    image, size = render_page(DocumentSource(path=path, sha256=sha256), page, dpi)
    return image, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/layout"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=256, help="pages rendered per GPU round")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from document_prep.heron import HeronLayoutDetector

    settings = PreparationSettings()
    detector = HeronLayoutDetector(threshold=0.3)
    pages = load_pages(args.benchmark)
    scale = 72 / settings.layout_dpi
    found: dict[str, dict[int, list[dict[str, object]]]] = defaultdict(dict)
    with ProcessPoolExecutor(args.workers) as pool:
        for start in range(0, len(pages), args.chunk):
            chunk = pages[start : start + args.chunk]
            rendered = list(pool.map(render, [(p.pdf, p.sha256, p.page, settings.layout_dpi) for p in chunk]))
            regions = detector.detect([image for image, _ in rendered])
            for page, page_regions in zip(chunk, regions, strict=True):
                found[page.document_id][page.page] = [
                    {
                        "label": r.label,
                        "score": r.score,
                        "box": [round(v * scale, 2) for v in (r.box.left, r.box.top, r.box.right, r.box.bottom)],
                    }
                    for r in page_regions
                ]
            print(f"{start + len(chunk)} / {len(pages)} pages", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for document_id, by_page in found.items():
        payload = {
            "model": detector.model_id,
            "threshold": detector.threshold,
            "dpi": settings.layout_dpi,
            "pages": by_page,
        }
        (args.out / f"{document_id}.json").write_text(json.dumps(payload) + "\n")
    print(f"{sum(len(v) for v in found.values())} pages in {len(found)} documents -> {args.out}")


if __name__ == "__main__":
    main()
