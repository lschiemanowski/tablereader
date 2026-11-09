"""Self-training on the rebuilt benchmark (user decision, 2026-11-08; planning/build-plan.md).

    uv run --group train python -m training.pipeline --run data/runs/r1

Stages, each writing into the run directory and skipped when its output exists:

- teacher: LightOnOCR-3-4B, 3 epochs on the accepted Terra labels of the training split.
- label1: the teacher transcribes every training part without a Terra label (~8,000).
- student1: from the base model, 1 epoch on the accepted self-labels plus the Terra training labels three times.
- polish: from the student, 1 epoch on the Terra training labels at half the learning rate.
- eval: every trained model on the validation parts, scored with the record metric against accepted Terra labels.

Training runs `training.sft` in a child process; labeling and evaluation serve each merged model with vLLM and
run the production extraction path (`extract_regions` with `VllmRecognizer`). The recipe follows ttr's.
"""

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.records import NOT_A_TABLE, RECORDS_VERSION, CropTables, crop_score
from extraction.extract import extract_regions
from extraction.results import RegionResult, RegionStatus, encode_region_result
from extraction.vllm import VllmRecognizer, VllmServer
from table_core.models import Table
from training.data import (
    BASE,
    MAX_SELF_LABEL_TOKENS,
    Recrops,
    Row,
    read_jsonl,
    region_id,
    reinterpret,
    rejection,
    training_target,
    write_jsonl,
)

STAGES = ("teacher", "label1", "student1", "polish", "eval")
TRAINED = ("teacher", "student1", "polish")


def crop_tables(result: RegionResult) -> CropTables:
    """A region's tables for the record metric: NOT_A_TABLE, the tables that have a candidate, or None."""
    if result.status is RegionStatus.NOT_A_TABLE:
        return NOT_A_TABLE
    if result.status is RegionStatus.FAILED:
        return None
    tables: list[Table] = [t.candidate for t in result.tables if t.candidate is not None]
    return tables or None


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dir: Path = args.run
        self.args = args
        self.recrops = Recrops.load(args.recrops)
        self.manifest = self.recrops.by_region()
        terra = read_jsonl(args.terra / "results.jsonl", partial_tail=True)
        self.terra = {r["region_id"]: r for r in terra}
        self.rng = random.Random(args.seed)
        self.terra_train, rejected = self.accepted(terra, "train")
        self.pool = [r for r in self.recrops.rows if r["split"] == "train" and region_id(r) not in self.terra]
        if args.limit_train:
            self.terra_train = self.terra_train[: args.limit_train]
        if args.limit_label:
            self.pool = self.pool[: args.limit_label]
        self.log(
            f"Terra training labels: {len(self.terra_train)} accepted, rejected {dict(rejected)}; "
            f"pool {len(self.pool)} training parts"
        )

    def accepted(self, labels: Sequence[Row], split: str) -> tuple[list[Row], Counter[str]]:
        kept: list[Row] = []
        rejected: Counter[str] = Counter()
        for label in labels:
            if label["split"] != split:
                continue
            if reason := rejection(label):
                rejected[reason] += 1
            else:
                kept.append(label)
        return kept, rejected

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.dir / "pipeline.log").open("a") as out:
            out.write(line + "\n")

    def model(self, stage: str) -> Path:
        return self.dir / stage / "merged"

    def name(self, stage: str) -> str:
        return f"{self.dir.name}-{stage}"

    def example(self, label: Row) -> Row:
        row = self.manifest[label["region_id"]]
        target = training_target(label["response"]["text"])
        return {"region_id": label["region_id"], "image": str(self.recrops.image_path(row)), "target": target}

    def train(self, stage: str, labels: Sequence[Row], *, base: str | Path, epochs: int, lr: float) -> None:
        if self.model(stage).exists():
            self.log(f"{stage}: model exists, skipped")
            return
        examples = [self.example(label) for label in labels]
        path = self.dir / f"{stage}.examples.jsonl"
        write_jsonl(path, examples)
        command = [
            sys.executable, "-m", "training.sft", "--examples", str(path), "--base", str(base),
            "--out", str(self.dir / stage), "--epochs", str(epochs), "--lr", str(lr), "--seed", str(self.args.seed),
        ]  # fmt: skip
        if self.args.max_steps:
            command += ["--max-steps", str(self.args.max_steps)]
        self.log(f"{stage}: training on {len(examples)} examples: {' '.join(command)}")
        with (self.dir / "logs" / f"sft_{stage}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    def transcribe(self, stage: str, rows: Sequence[Row], out: Path) -> list[Row]:
        """Every row's label from `stage`'s model, in `out` (appended as results arrive; done rows skipped)."""
        done = {r["region_id"] for r in read_jsonl(out)}
        todo = [r for r in rows if region_id(r) not in done]
        if todo:
            self.log(f"{stage}: transcribing {len(todo)} parts into {out.name} ({len(done)} already done)")
            log = self.dir / "logs" / f"vllm_{stage}.log"
            with VllmServer(self.model(stage), name=self.name(stage), log=log) as server:
                asyncio.run(self._transcribe(server, stage, todo, out))
        wanted = {region_id(r) for r in rows}
        return [r for r in read_jsonl(out) if r["region_id"] in wanted]

    async def _transcribe(self, server: VllmServer, stage: str, rows: Sequence[Row], out: Path) -> None:
        recognizer = VllmRecognizer(model=server.name, base_url=server.base_url)
        by_region = {region_id(r): r for r in rows}
        statuses: Counter[str] = Counter()
        started = time.monotonic()
        try:
            with out.open("a") as sink:
                regions = self.recrops.regions(rows)
                async for result in extract_regions(regions, recognizer, max_in_flight=self.args.max_in_flight):
                    row = by_region[result.region_id]
                    payload = {
                        "region_id": result.region_id, "table_id": row["table_id"], "part": row["part"],
                        "split": row["split"], "image_sha256": row["sha256"], **encode_region_result(result),
                    }  # fmt: skip
                    sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    sink.flush()
                    statuses[result.status.value] += 1
                    n = statuses.total()
                    if n % 200 == 0 or n == len(rows):
                        rate = (time.monotonic() - started) / n
                        self.log(f"{stage}: {n}/{len(rows)} {dict(statuses)} {rate:.2f} s/part")
        finally:
            await recognizer.aclose()

    def stage(self, name: str) -> None:
        started = time.monotonic()
        self.log(f"== {name}")
        lr = self.args.lr
        if name == "teacher":
            self.train("teacher", self.terra_train, base=BASE, epochs=3, lr=lr)
        elif name == "label1":
            self.transcribe("teacher", self.pool, self.dir / "label1.jsonl")
        elif name == "student1":
            labels = self.transcribe("teacher", self.pool, self.dir / "label1.jsonl")
            kept: list[Row] = []
            why: Counter[str] = Counter()
            for label in labels:
                if reason := rejection(label, max_tokens=MAX_SELF_LABEL_TOKENS):
                    why[reason] += 1
                else:
                    kept.append(label)
            self.log(f"student1: {len(kept)} of {len(labels)} self-labels kept; rejected {dict(why)}")
            mixed = kept + self.terra_train * 3
            self.rng.shuffle(mixed)
            self.train("student1", mixed, base=BASE, epochs=1, lr=lr)
        elif name == "polish":
            self.train("polish", self.terra_train, base=self.model("student1"), epochs=1, lr=lr / 2)
        elif name == "eval":
            self.evaluate()
        else:
            raise SystemExit(f"unknown stage {name}; stages are {', '.join(STAGES)}")
        self.log(f"== {name} done in {(time.monotonic() - started) / 60:.0f} min")

    def evaluate(self) -> None:
        # read now, not at start: validation labeling may still have been running when the pipeline began
        validation, rejected = self.accepted(
            read_jsonl(self.args.terra / "results.jsonl", partial_tail=True), "validation"
        )
        if self.args.limit_eval:
            validation = validation[: self.args.limit_eval]
        expected = sum(r["split"] == "validation" for r in self.recrops.rows)
        self.log(
            f"eval: {len(validation)} accepted Terra validation labels of {expected} parts; rejected {dict(rejected)}"
        )
        rows = [self.manifest[label["region_id"]] for label in validation]
        references = {label["region_id"]: crop_tables(reinterpret(label)) for label in validation}
        report: dict[str, Any] = {
            "metric": RECORDS_VERSION, "reference": "accepted Terra validation labels", "parts": len(validation),
            "rejected_references": dict(rejected), "validation_parts": expected,
        }  # fmt: skip
        for stage in TRAINED:
            if not self.model(stage).exists():
                continue
            predictions = self.transcribe(stage, rows, self.dir / f"eval_{stage}.jsonl")
            scores: dict[str, list[float]] = defaultdict(list)
            outcomes: Counter[str] = Counter()
            for label in predictions:
                result = reinterpret(label) if label["response"] is not None else None
                predicted = crop_tables(result) if result is not None else None
                score = crop_score(references[label["region_id"]], predicted)
                medium = self.manifest[label["region_id"]]["medium"]
                for group in ("all", medium):
                    scores[group].append(score)
                outcomes[label["status"]] += 1
                for table in label["tables"]:
                    outcomes[f"table {table['outcome']}"] += 1
            report[stage] = {
                "parts": len(predictions),
                "outcomes": dict(outcomes),
                "records": {group: round(mean(values), 4) for group, values in sorted(scores.items())},
            }
            self.log(f"eval {stage}: {json.dumps(report[stage])}")
        (self.dir / "eval.json").write_text(json.dumps(report, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory, e.g. data/runs/r1")
    parser.add_argument("--recrops", type=Path, default=Path("data/recrops"))
    parser.add_argument("--terra", type=Path, default=Path("data/labels/terra"))
    parser.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-in-flight", type=int, default=64)
    parser.add_argument("--limit-train", type=int, help="smoke tests: at most N Terra training labels")
    parser.add_argument("--limit-label", type=int, help="smoke tests: at most N pool parts")
    parser.add_argument("--limit-eval", type=int, help="smoke tests: at most N validation parts")
    parser.add_argument("--max-steps", type=int, help="smoke tests: optimizer steps per training stage")
    args = parser.parse_args(argv)
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "logs").mkdir(exist_ok=True)
    pipeline = Pipeline(args)
    pipeline.log(f"run {args.run} with {json.dumps({k: str(v) for k, v in vars(args).items()})}")
    for stage in args.only or STAGES:
        pipeline.stage(stage)


if __name__ == "__main__":
    main()
