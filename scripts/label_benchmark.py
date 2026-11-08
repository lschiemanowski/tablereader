"""Label benchmark recrops with GPT-6.1 Terra through the headless Codex CLI.

Scope (user decision, 2026-11-08): every validation and test part, and the parts of the 1,000 training tables
in table_benchmark's earlier Terra cohort (`data/full/terra-6.1-train/cohort.jsonl`), so the new labels can be
cross-checked against both v1 and the earlier Terra grids. Reads the recrops from `recrop_benchmark.py` and
appends one JSON line per part to `<out>/results.jsonl` as results arrive; a rerun skips parts already done.

    uv run python scripts/label_benchmark.py ~/workspace/table_benchmark --pilot 40   # stratified sample
    uv run python scripts/label_benchmark.py ~/workspace/table_benchmark              # full scope
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from document_prep.models import Box, DocumentSource, PreparationSettings, PreparedRegion, TableRegion
from extraction.codex import DEFAULT_CLI, PROMPT_SHA256, CodexRecognizer
from extraction.extract import extract_regions
from extraction.results import encode_region_result


def _box(values: list[float]) -> Box:
    return Box(left=values[0], top=values[1], right=values[2], bottom=values[3])


def prepared(row: dict[str, Any], recrops: Path, settings: PreparationSettings, renderer: str) -> PreparedRegion:
    """A PreparedRegion rebuilt from a recrop manifest line and its saved PNG."""
    image = Image.open(recrops / row["image"])
    image.load()
    return PreparedRegion(
        source=DocumentSource(path=Path(row["document_id"]), sha256=""),
        region=TableRegion(
            region_id=Path(row["image"]).stem, page=row["page"], box=_box(row["detected_box"]),
            rotation_cw=row["rotation_cw"],
        ),
        page_size=(row["page_size"][0], row["page_size"][1]),
        page_pixels=(row["page_pixels"][0], row["page_pixels"][1]),
        detected_px=_box(row["detected_px"]),
        crop_px=_box(row["crop_px"]),
        crop_box=_box(row["crop_box"]),
        image=image,
        settings=settings,
        renderer=renderer,
    )  # fmt: skip


def select(rows: list[dict[str, Any]], benchmark: Path, pilot: int | None, seed: int) -> list[dict[str, Any]]:
    cohort_path = benchmark / "data/full/terra-6.1-train/cohort.jsonl"
    cohort = {json.loads(line)["crop_key"] for line in cohort_path.open()}
    chosen = [r for r in rows if r["split"] != "train" or r["crop_key"] in cohort]
    if pilot is None:
        return chosen
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in chosen:
        strata[(r["split"], r["medium"])].append(r)
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for key in sorted(strata):
        quota = max(1, round(pilot * len(strata[key]) / len(chosen)))
        sample.extend(rng.sample(strata[key], min(quota, len(strata[key]))))
    rng.shuffle(sample)
    return sample[:pilot]


async def run(args: argparse.Namespace) -> None:
    header = json.loads((args.recrops / "settings.json").read_text())
    fields: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in header["settings"].items()}
    settings = PreparationSettings(**fields)
    rows = [json.loads(line) for line in (args.recrops / "manifest.jsonl").open()]
    todo = select(rows, args.benchmark, args.pilot, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    done = {json.loads(line)["region_id"] for line in results_path.open()} if results_path.exists() else set[str]()
    todo = [r for r in todo if Path(r["image"]).stem not in done]
    by_region = {Path(r["image"]).stem: r for r in todo}
    print(f"{len(todo)} parts to label ({len(done)} already done); prompt {PROMPT_SHA256[:12]}", flush=True)

    def regions() -> Iterator[PreparedRegion]:
        for r in todo:
            yield prepared(r, args.recrops, settings, header["renderer"])

    recognizer = CodexRecognizer(cli=args.cli, model=args.model, reasoning=args.reasoning)
    statuses: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    start = time.monotonic()
    with results_path.open("a") as out:
        async for result in extract_regions(regions(), recognizer, max_in_flight=args.max_in_flight):
            row = by_region[result.region_id]
            payload = {
                "region_id": result.region_id,
                "table_id": row["table_id"],
                "part": row["part"],
                "split": row["split"],
                "v1_label": row["label"],
                "image_sha256": row["sha256"],
                **encode_region_result(result),
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out.flush()
            statuses[result.status.value] += 1
            for t in result.tables:
                statuses[f"table {t.outcome.value}"] += 1
            if result.response is not None:
                usage.update(result.response.usage)
            n = sum(statuses[s] for s in ("tables", "not_a_table", "failed"))
            if n % 10 == 0 or n == len(todo):
                rate = (time.monotonic() - start) / n
                print(f"{n}/{len(todo)} {dict(statuses)} {rate:.1f} s/part", flush=True)
    summary = {
        "model": args.model,
        "reasoning": args.reasoning,
        "prompt_sha256": PROMPT_SHA256,
        "parts": len(todo),
        "statuses": dict(statuses),
        "usage": dict(usage),
        "seconds": round(time.monotonic() - start, 1),
    }
    with (args.out / "runs.jsonl").open("a") as log:
        log.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--recrops", type=Path, default=Path("data/recrops"))
    parser.add_argument("--out", type=Path, default=Path("data/labels/terra"))
    parser.add_argument("--pilot", type=int, help="label a stratified sample of this size instead of the full scope")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--max-in-flight", type=int, default=6)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--model", default="gpt-6.1-terra")
    parser.add_argument("--reasoning", default="low")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
