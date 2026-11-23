"""Pilot of a fine-tuned structure reviewer: is Gemma 4 12B, trained on verified DeepSeek answers, worth the full build?

The drafts are r4-vision's validation drafts (real first-pass errors; the OCR model never trained on them), split
by document: the evaluation side is the review trial's crops (`data/reviews/haiku-shift/sample.jsonl`) in the held-out
documents, so every earlier reviewer can be scored on them for free; the training side is every reviewable crop in
the other documents. Steps:

    uv run python scripts/review_pilot.py split       # documents, eval.jsonl and train.jsonl in data/reviews/pilot
    uv run python scripts/review_benchmark.py --crops data/reviews/pilot/train.jsonl --samples 2 \\
        --model deepseek/deepseek-v4.1-flash --quantization fp8 --provider-sort price --max-tokens 64000 \\
        --out data/reviews/pilot-labels                # DeepSeek labels (OpenRouter)
    uv run python scripts/review_pilot.py examples --max-reasoning-tokens 16000 --out data/reviews/pilot/cap16k
    python -m training.review_sft ...                  # see training.review_sft

The second pilot (`split2`, in data/reviews/pilot2) keeps the first one's crops and eval crops and adds the hard
crops it lacked, outside the eval documents: drafts of older checkpoints on validation crops (the newest such
draft per crop), and the self-training teachers' first-pass drafts of training crops they never trained on
(`label1.jsonl`, r2's before r1's), with at most `--treasury` US Treasury Bulletin crops. Each needs an accepted
Terra reference (`review_benchmark.terra_labels`). `new.jsonl` lists the added crops for labeling; `examples
--pilot data/reviews/pilot2 --labels data/reviews/pilot-labels --labels data/reviews/pilot2-labels` combines both.

`examples` keeps, per crop, the best answer that passes the review guards and does not lower the crop's records-2
score against its Terra reference (ties go to a kept draft, then to the shorter reasoning), and drops answers
whose reasoning exceeds `--max-reasoning-tokens`. It writes the examples with reasoning, the same answers without
it, a half of the crops (half of each kind) for the learning-curve comparison, and a split with 15% of each kind
held out (`sweep-train.jsonl`, `sweep-dev.jsonl`) for choosing the learning rate by held-out loss.
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from review_benchmark import (
    RUN,
    SOURCES,
    answers_of,
    candidates,
    draft_files,
    drafted_rows,
    ready,
    references_and_predictions,
    tables_with,
    terra_labels,
)

from evaluation.records import crop_score
from evaluation.review import OutcomeState, judge, split_draft
from training.data import read_jsonl, rejection

PILOT = Path("data/reviews/pilot")
PILOT2 = Path("data/reviews/pilot2")
OLDER = ("r4-rank64", "r4-epoch1", "r3", "r2", "r1")  # checkpoints before r4-vision, newest first
TREASURY = "us-treasury-bulletin"
TRIAL = Path("data/reviews/haiku-shift/sample.jsonl")
HARD = "scan, draft not ready"


def manifest() -> dict[str, dict[str, Any]]:
    """Crop id -> its recrop manifest line."""
    out: dict[str, dict[str, Any]] = {}
    for recrops, _, _ in SOURCES.values():
        for row in read_jsonl(Path(recrops) / "manifest.jsonl"):
            out[Path(row["image"]).stem] = row
    return out


def documents() -> dict[str, str]:
    """Crop id -> document id, from the recrop manifests."""
    return {rid: row["document_id"] for rid, row in manifest().items()}


def choose_eval_documents(rows: list[dict[str, Any]], trial: set[str], doc: dict[str, str]) -> set[str]:
    """Documents to hold out: about 25 of the trial's hard crops in them, as many hard crops as possible elsewhere.
    A fixed search over random halvings, so the split is reproducible."""
    by: dict[str, Counter[tuple[str, bool]]] = defaultdict(Counter)
    for r in rows:
        side = "trial" if r["region_id"] in trial else "pool"
        by[doc[r["region_id"]]][(side, r["stratum"] == HARD)] += 1
    names = sorted(by)
    best: tuple[float, set[str]] | None = None
    for seed in range(20000):
        rng = random.Random(seed)
        held = {d for d in names if rng.random() < 0.5}
        eval_hard = sum(by[d][("trial", True)] for d in held)
        train_hard = sum(by[d][("trial", True)] + by[d][("pool", True)] for d in names if d not in held)
        value = min(eval_hard, 25) + min(train_hard, 60) - abs(eval_hard - 25)
        if best is None or value > best[0]:
            best = (value, held)
    assert best is not None
    return best[1]


def split(ready_crops: int, seed: int) -> None:
    rows = candidates(RUN)
    trial = {r["region_id"] for r in read_jsonl(TRIAL)}
    doc = documents()
    held = choose_eval_documents(rows, trial, doc)
    evaluation = [r for r in rows if r["region_id"] in trial and doc[r["region_id"]] in held]
    train_side = [r for r in rows if doc[r["region_id"]] not in held]
    hard = [r for r in train_side if r["stratum"] == HARD]
    ready = [r for r in train_side if r["stratum"] != HARD]
    # half of them drafts with errors (they may hold a structure fix), half clean ones to keep
    references, predictions = references_and_predictions({r["region_id"] for r in ready})
    flawed = [
        r for r in ready if crop_score(references[r["region_id"]], tables_with(predictions[r["region_id"]])) < 0.95
    ]
    clean = [r for r in ready if r not in flawed]
    rng = random.Random(seed)
    some_flawed = rng.sample(flawed, min(len(flawed), ready_crops // 2))
    chosen = hard + some_flawed + rng.sample(clean, min(len(clean), ready_crops - len(some_flawed)))
    PILOT.mkdir(parents=True, exist_ok=True)
    for name, part in (("eval", evaluation), ("train", chosen)):
        lines = [json.dumps({"region_id": r["region_id"], "stratum": r["stratum"]}) + "\n" for r in part]
        (PILOT / f"{name}.jsonl").write_text("".join(lines))
    split_info = {
        "eval_documents": sorted(held),
        "train_documents": sorted({doc[r["region_id"]] for r in train_side}),
        "eval": dict(Counter(r["stratum"] for r in evaluation)),
        "train": dict(Counter(r["stratum"] for r in chosen)),
        "train_ready_flawed": len(some_flawed),
        "seed": seed,
    }
    (PILOT / "split.json").write_text(json.dumps(split_info, indent=2) + "\n")
    print(json.dumps(split_info, indent=2))


def hard_drafts(file: Path, eligible: set[str]) -> list[str]:
    """Crops of `eligible` whose draft in this prediction file has a table that is not ready."""
    return [
        p["region_id"]
        for p in read_jsonl(file)
        if p["region_id"] in eligible and p.get("response") and not ready(p) and split_draft(p["response"]["text"])
    ]


def split2(treasury: int, seed: int) -> None:
    first = read_jsonl(PILOT / "train.jsonl")
    held = set(json.loads((PILOT / "split.json").read_text())["eval_documents"])
    rows = manifest()
    referenced = {label["region_id"]: label["split"] for label in terra_labels() if rejection(label) is None}
    eligible = {
        rid for rid in referenced
        if rid in rows and rows[rid]["medium"] != "born_digital" and rows[rid]["document_id"] not in held
    } - {r["region_id"] for r in first}  # fmt: skip
    older: dict[str, Path] = {}
    for run in OLDER:
        for file in sorted(Path("data/runs", run).glob("*.jsonl")):
            if file.name.startswith(("eval_", "scans-validation_")):
                for rid in hard_drafts(file, {r for r in eligible if referenced[r] == "validation"} - set(older)):
                    older[rid] = file
    unlabelled: dict[str, Path] = {}
    for run in ("r2", "r1"):
        file = Path("data/runs", run, "label1.jsonl")
        for rid in hard_drafts(file, eligible - set(older) - set(unlabelled)):
            unlabelled[rid] = file
    rng = random.Random(seed)
    bulletin = sorted(r for r in unlabelled if rows[r]["source_id"] == TREASURY)
    dropped = set(bulletin) - set(rng.sample(bulletin, min(treasury, len(bulletin))))
    added = {**older, **{r: f for r, f in unlabelled.items() if r not in dropped}}
    new = [{"region_id": r, "stratum": HARD, "drafts": str(f)} for r, f in sorted(added.items())]
    PILOT2.mkdir(parents=True, exist_ok=True)
    write(PILOT2 / "new.jsonl", new)
    write(PILOT2 / "train.jsonl", [*first, *new])
    shutil.copy(PILOT / "eval.jsonl", PILOT2 / "eval.jsonl")
    info = {
        "first_pilot_crops": len(first),
        "added": len(new),
        "from_older_checkpoints": len(older),
        "from_label1": len(added) - len(older),
        "treasury_dropped": len(dropped),
        "added_by_file": dict(Counter(str(f) for f in added.values())),
        "added_by_series": dict(Counter(rows[r]["source_id"] for r in added).most_common()),
        "added_documents": len({rows[r]["document_id"] for r in added}),
        "seed": seed,
    }
    (PILOT2 / "split.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


def reasoning_tokens(answer: dict[str, Any]) -> int:
    return int(answer.get("usage", {}).get("completion_tokens_details.reasoning_tokens", 0))


def examples(pilot: Path, labels: list[Path], max_reasoning_tokens: int, seed: int, out: Path) -> None:
    lines = read_jsonl(pilot / "train.jsonl")
    train = {r["region_id"]: r["stratum"] for r in lines}
    records = [r for d in labels for r in read_jsonl(d / "results.jsonl") if r["region_id"] in train]
    files = draft_files(lines)
    references, predictions = references_and_predictions(set(train), drafts=files)
    rows = {r["region_id"]: r for r in candidates(RUN) if r["region_id"] in train and r["region_id"] not in files}
    rows |= {r["region_id"]: r for r in drafted_rows(lines)}
    kept: list[dict[str, Any]] = []
    why: Counter[str] = Counter()
    for record in records:
        rid = record["region_id"]
        draft = rows[rid]["draft"]
        before = crop_score(references[rid], tables_with(predictions[rid]))
        options: list[tuple[tuple[float, int, int], dict[str, Any], str]] = []
        for stored, answer in zip(record["answers"], answers_of(record), strict=True):
            if answer is None:
                why["no answer"] += 1
                continue
            if reasoning_tokens(stored) > max_reasoning_tokens:
                why["reasoning too long"] += 1
                continue
            outcome = judge(rid, draft, answer)
            if outcome.state is OutcomeState.REJECTED:
                why["rejected by the guards"] += 1
                continue
            after = before
            if outcome.state is OutcomeState.REVISED:
                after = crop_score(references[rid], tables_with(predictions[rid], outcome.text))
            if after < before - 1e-9:
                why["lowers the score"] += 1
                continue
            rank = (after, outcome.state is not OutcomeState.REVISED, -reasoning_tokens(stored))
            options.append((rank, stored, outcome.state.value))
        if not options:
            why["crop without a usable answer"] += 1
            continue
        (after, _, _), best, state = max(options, key=lambda o: o[0])
        answer_text = json.dumps({"verdict": best["verdict"], "edits": best["edits"]}, ensure_ascii=False)
        kept.append({
            "region_id": rid, "stratum": train[rid], "image": rows[rid]["image"], "draft": draft,
            "reasoning": best.get("reasoning") or "", "answer": answer_text, "state": state,
            "gain": round(after - before, 4),
        })  # fmt: skip
    rng = random.Random(seed)
    half: list[dict[str, Any]] = []
    for kind in (True, False):  # half of the hard crops and half of the ready ones
        group = [e for e in kept if (e["stratum"] == HARD) is kind]
        half += rng.sample(group, len(group) // 2)
    dev: list[dict[str, Any]] = []
    for kind in (True, False):  # 15% of each kind held out to choose the learning rate
        group = [e for e in kept if (e["stratum"] == HARD) is kind]
        dev += rng.sample(group, round(0.15 * len(group)))
    out.mkdir(parents=True, exist_ok=True)
    write(out / "examples-reasoning.jsonl", kept)
    write(out / "examples-reasoning-half.jsonl", half)
    write(out / "examples-answer.jsonl", [dict(e, reasoning="") for e in kept])
    write(out / "sweep-train.jsonl", [e for e in kept if e not in dev])
    write(out / "sweep-dev.jsonl", dev)
    print(f"{len(kept)} examples from {len(records)} labelled crops ({len(train)} chosen); dropped: {dict(why)}")
    print("by stratum and outcome:", dict(Counter((e["stratum"], e["state"]) for e in kept)))
    gains = [e["gain"] for e in kept]
    print(f"improving: {sum(g > 1e-9 for g in gains)}, mean gain {sum(gains) / max(1, len(gains)):+.3f}")


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    s = sub.add_parser("split", help="choose the held-out documents and the training crops")
    s.add_argument("--ready-crops", type=int, default=200, help="ready drafts to label besides the hard ones")
    s.add_argument("--seed", type=int, default=20261121)
    s2 = sub.add_parser("split2", help="the second pilot: the first one's crops plus the hard crops it lacked")
    s2.add_argument("--treasury", type=int, default=60, help="at most this many US Treasury Bulletin crops")
    s2.add_argument("--seed", type=int, default=20261122)
    e = sub.add_parser("examples", help="turn verified labels into fine-tuning examples")
    e.add_argument("--pilot", type=Path, default=PILOT, help="folder with the pilot's train.jsonl")
    e.add_argument("--labels", type=Path, action="append", help="label folders (default: data/reviews/pilot-labels)")
    e.add_argument("--max-reasoning-tokens", type=int, default=12000)
    e.add_argument("--out", type=Path, default=PILOT, help="folder for the example files")
    e.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    if args.step == "split":
        split(args.ready_crops, args.seed)
    elif args.step == "split2":
        split2(args.treasury, args.seed)
    else:
        labels = args.labels or [Path("data/reviews/pilot-labels")]
        examples(args.pilot, labels, args.max_reasoning_tokens, args.seed, args.out)


if __name__ == "__main__":
    main()
