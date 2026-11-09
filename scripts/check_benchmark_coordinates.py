"""Check that document_prep places benchmark boxes where the v1 crops were cut.

Renders every page at 180 DPI (the v1 resolution) with document_prep, cuts each part's tight box, and compares
it with the v1 PNG (rendered by Poppler). Reports size mismatches and the best alignment within a few pixels.

    uv run python scripts/check_benchmark_coordinates.py path/to/table_benchmark --workers 32
"""

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_regions import Page, load_pages

from document_prep.models import DocumentSource
from document_prep.prepare import render_page

V1_DPI = 180


def check_page(args: tuple[Page, Path]) -> list[dict[str, object]]:
    page, release = args
    source = DocumentSource(path=page.pdf, sha256=page.sha256)
    image, _ = render_page(source, page.page, V1_DPI)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    scale = V1_DPI / 72
    results: list[dict[str, object]] = []
    for part in page.parts:
        box = part.region.box.scaled(scale)
        x, y = max(0, math.floor(box.left)), max(0, math.floor(box.top))
        ref = np.asarray(Image.open(release / part.part["image"]).convert("L"), dtype=np.float32)
        h, w = ref.shape
        want = (math.ceil(box.right) - x, math.ceil(box.bottom) - y)
        best = (math.inf, 0, 0)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if x + dx < 0 or y + dy < 0:
                    continue
                cut = gray[y + dy : y + dy + h, x + dx : x + dx + w]
                if cut.shape == ref.shape:
                    best = min(best, (float(np.abs(cut - ref).mean()), dx, dy))
        results.append(
            {
                "region_id": part.region.region_id,
                "size_matches": want == (w, h),
                "mean_abs": best[0],
                "shift": [best[1], best[2]],
                "page_rotation_or_rotated": part.region.rotation_cw,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/coordinate-check.jsonl"))
    args = parser.parse_args()
    release = args.benchmark / "data/full/release-v1"
    pages = load_pages(args.benchmark)
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(args.workers) as pool:
        for rows in pool.map(check_page, [(p, release) for p in pages], chunksize=8):
            results.extend(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in results))
    sizes = Counter(bool(r["size_matches"]) for r in results)
    shifts = Counter(tuple(r["shift"]) for r in results)  # type: ignore[arg-type]
    diffs = np.array([r["mean_abs"] for r in results], dtype=float)
    print(f"parts {len(results)}; size matches {sizes[True]}, differs {sizes[False]}")
    print("best shift:", shifts.most_common(6))
    print("mean abs difference (0-255): " + ", ".join(f"p{q} {np.percentile(diffs, q):.1f}" for q in (50, 90, 99, 100)))
    worst = sorted(results, key=lambda r: -float(r["mean_abs"]))[:8]  # type: ignore[arg-type]
    for r in worst:
        print("  worst:", r)


if __name__ == "__main__":
    main()
