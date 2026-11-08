"""Cross-check new transcription labels against table_benchmark's v1 grids and its earlier Terra grids.

For every labeled part, the transcription becomes a grid (title and footnotes dropped, markers put back) and is
compared with the v1 reference grid of the same crop part, and with the earlier Terra grid where one exists.
Writes `<labels>/crosscheck.jsonl` and prints agreement by split and medium. Read-only on the benchmark.

    uv run python scripts/crosscheck_labels.py ~/workspace/table_benchmark --labels data/labels/terra
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from evaluation.grids import Grid, GridCell, compare_grids, grid_from_parsed
from table_core.parsing import ResponseKind, parse_output


def reference_grids(path: Path) -> dict[tuple[str, int], Grid]:
    grids: dict[tuple[str, int], Grid] = {}
    if not path.exists():
        return grids
    for line in path.open():
        row = json.loads(line)
        for part in row.get("parts", []):
            cells = tuple(
                GridCell(c["row"], c["column"], c["row_span"], c["column_span"], c["text"]) for c in part["cells"]
            )
            grids[(row["crop_key"], part["part"])] = Grid(rows=part["rows"], columns=part["columns"], cells=cells)
    return grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("data/labels/terra"))
    parser.add_argument("--recrops", type=Path, default=Path("data/recrops"))
    args = parser.parse_args()
    full = args.benchmark / "data/full"
    v1 = reference_grids(full / "release-v1/cell-references.jsonl")
    terra: dict[tuple[str, int], Grid] = {}
    for split, name in (("train", "train.jsonl"), ("validation", "validation.jsonl"), ("test", "test.jsonl")):
        terra |= reference_grids(full / f"terra-6.1-{split}" / name)
    manifest = {Path(r["image"]).stem: r for r in map(json.loads, (args.recrops / "manifest.jsonl").open())}

    rows: list[dict[str, Any]] = []
    for line in (args.labels / "results.jsonl").open():
        label = json.loads(line)
        m = manifest[label["region_id"]]
        key = (m["crop_key"], m["part"])
        row: dict[str, Any] = {
            "region_id": label["region_id"],
            "table_id": label["table_id"],
            "split": m["split"],
            "medium": m["medium"],
            "v1_label": m["label"],
            "status": label["status"],
            "outcomes": [t["outcome"] for t in label["tables"]],
        }
        if label["status"] == "tables":
            output = parse_output(label["response"]["text"])
            if output.kind is ResponseKind.TABLES:
                grid = grid_from_parsed([f.table for f in output.fragments if f.table is not None])
                for name, references in (("v1", v1), ("terra", terra)):
                    if key in references:
                        c = compare_grids(references[key], grid)
                        row[name] = {
                            "same_shape": c.same_shape,
                            "structure_f1": round(c.structure_f1, 4),
                            "text_f1": round(c.text_f1, 4),
                            "content_f1": round(c.content_f1, 4),
                        }
        rows.append(row)
    (args.labels / "crosscheck.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    print(f"{len(rows)} parts")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["split"]].append(r)
        groups[r["medium"]].append(r)
    for name, members in groups.items():
        print(f"\n{name}: {len(members)} parts")
        agree = sum((r["status"] == "not_a_table") == (r["v1_label"] == "not_a_table") for r in members)
        print(f"  non-table decision agrees with v1: {agree}/{len(members)}")
        for ref in ("v1", "terra"):
            scored = [r[ref] for r in members if ref in r]
            if not scored:
                continue
            content = sorted(s["content_f1"] for s in scored)
            print(
                f"  vs {ref:5} n={len(scored):5}  same shape {mean(s['same_shape'] for s in scored):.2f}"
                f"  structure {median(s['structure_f1'] for s in scored):.3f}"
                f"  text {median(s['text_f1'] for s in scored):.3f}"
                f"  content median {median(content):.3f}  <0.9: {sum(c < 0.9 for c in content)}"
                f"  <0.7: {sum(c < 0.7 for c in content)}"
            )


if __name__ == "__main__":
    main()
