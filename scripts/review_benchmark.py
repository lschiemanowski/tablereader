"""Trial of LLM review (evaluation.review) on first-pass transcriptions of benchmark validation crops.

Draws a stratified sample of crops with an accepted Terra reference from a run's validation predictions (v1 and
the scan expansion): scanned crops whose draft has a non-ready table, scanned crops whose draft is ready, and
born-digital crops whose draft is ready. Each draft is reviewed once through OpenRouter (key in
OPENROUTER_API_KEY); results are appended to `<out>/results.jsonl` (a rerun skips crops already done) and scored
against the references with records-2. The saved drafts are used, so no GPU is needed.

    uv run python scripts/review_benchmark.py --dry-run          # sample and a rough cost estimate; no API calls
    uv run python scripts/review_benchmark.py                    # review the sample
    uv run python scripts/review_benchmark.py --score            # score what is there

`--crops FILE` (JSON lines with `region_id`) reviews, or scores, those crops instead of the stratified sample;
the review pilot (`scripts/review_pilot.py`) uses it for its training and evaluation crops. A line may also name
`drafts` (a prediction file, e.g. a self-training run's `label1.jsonl`) and `stratum`: the crop's draft is then
taken from that file instead of the run's validation predictions, and its reference may also come from
`EXTRA_TERRA` (Terra labels made for such crops).
"""

import argparse
import asyncio
import dataclasses
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

from evaluation.records import crop_score
from evaluation.review import (
    OPENROUTER_URL,
    Edit,
    EditOp,
    OpenRouterReviewer,
    OutcomeState,
    ReviewAnswer,
    ReviewRequest,
    Verdict,
    agree,
    judge,
    render_view,
    split_draft,
)
from extraction.results import ComponentFailure
from extraction.vllm import model_image
from training.data import read_jsonl, reinterpret, rejection
from training.pipeline import crop_tables

SOURCES = {  # name -> (recrops, Terra labels, prediction file in the run directory)
    "v1": ("data/recrops", "data/labels/terra", "eval_teacher.jsonl"),
    "scans": ("data/recrops-scan-expansion", "data/labels/terra-scan-expansion", "scans-validation_teacher.jsonl"),
}
STRATA = ("scan, draft not ready", "scan, draft ready", "digital, draft ready")
RUN = Path("data/runs/r4-vision")  # the first-pass drafts reviewed
EXTRA_TERRA = (Path("data/labels/terra-review2"),)  # references for training crops reviewed in the second pilot
PRICE = {
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "google/gemma-4-26b-a4b-it": (0.10, 0.30),
    "deepseek/deepseek-v4.1-flash": (0.05, 0.20),
    "z-ai/glm-5.3-flash": (0.15, 0.50),
}  # $ per million input / output tokens (OpenRouter list price)


def ready(prediction: dict[str, Any]) -> bool:
    return prediction["status"] == "tables" and all(t["outcome"] == "ready" for t in prediction["tables"])


def candidates(run: Path) -> list[dict[str, Any]]:
    """Validation crops with an accepted reference and a reviewable draft, with their stratum."""
    out: list[dict[str, Any]] = []
    for source, (recrops, terra, predictions) in SOURCES.items():
        manifest = {Path(r["image"]).stem: r for r in read_jsonl(Path(recrops) / "manifest.jsonl")}
        accepted = {
            label["region_id"]
            for label in read_jsonl(Path(terra) / "results.jsonl")
            if label["split"] == "validation" and rejection(label) is None
        }
        for p in read_jsonl(run / predictions):
            if p["region_id"] not in accepted or p["response"] is None or split_draft(p["response"]["text"]) is None:
                continue
            row = manifest[p["region_id"]]
            scanned = row["medium"] != "born_digital"
            stratum = ("scan" if scanned else "digital") + (", draft ready" if ready(p) else ", draft not ready")
            if stratum in STRATA:
                image = str(Path(recrops) / row["image"])
                out.append({"region_id": p["region_id"], "source": source, "stratum": stratum, "image": image,
                            "series": row["source_id"], "draft": p["response"]["text"]})  # fmt: skip
    return out


def sample(rows: list[dict[str, Any]], per_stratum: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for stratum in STRATA:
        pool = [r for r in rows if r["stratum"] == stratum]
        chosen += rng.sample(pool, min(per_stratum, len(pool)))
    rng.shuffle(chosen)
    return chosen


def estimate(rows: list[dict[str, Any]], reviewer: OpenRouterReviewer, output_tokens: int, samples: int) -> None:
    """A rough cost estimate: text at 3.2 characters per token, images at width x height / 750 (capped)."""
    prompt = len(
        reviewer.request(ReviewRequest("x", Image.new("RGB", (8, 8)), rows[0]["draft"]))["messages"][0]["content"][0][
            "text"
        ]
    )
    text = image = 0.0
    for r in rows:
        with Image.open(r["image"]) as im:
            w, h = model_image(im, reviewer.max_edge).size
        image += min(w * h / 750, 4784)
        text += len(render_view(r["draft"])) / 3.2
    n = len(rows) * samples
    image *= samples
    text *= samples
    inp, out = PRICE.get(reviewer.model, (0.0, 0.0))
    cached = prompt / 3.2 * (1.25 + 0.1 * (n - 1))  # one cache write, then reads
    total_in = text + image + cached
    cost = (total_in * inp + n * output_tokens * out) / 1e6
    print(f"{n} reviews: prompt ~{prompt / 3.2:.0f} tokens (cached), views ~{text / n:.0f} and images ~{image / n:.0f}")
    print(f"tokens per crop; assuming {output_tokens} output tokens per answer: about ${cost:.2f} in total")


async def review_once(reviewer: OpenRouterReviewer, request: ReviewRequest) -> dict[str, Any]:
    """One review as a JSON record, retrying retryable failures twice; an `error` record if it still fails."""
    for attempt in range(3):
        try:
            answer = await reviewer.review(request)
        except ComponentFailure as failure:
            if not failure.retryable or attempt == 2:
                return {"error": f"{failure.code}: {failure.message}"}
            await asyncio.sleep(5 * 2**attempt)
        else:
            return {"verdict": answer.verdict.value, "edits": json.loads(answer.raw)["edits"], "usage": answer.usage,
                    "cost": answer.cost, "model": answer.model, "settings": answer.settings,
                    "reasoning": answer.reasoning}  # fmt: skip
    raise AssertionError("unreachable")


async def review_all(
    rows: list[dict[str, Any]], reviewer: OpenRouterReviewer, out: Path, concurrency: int, samples: int
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    started = time.monotonic()

    async def one(row: dict[str, Any], sink: Any) -> None:
        nonlocal done
        with Image.open(row["image"]) as image:
            image.load()
            request = ReviewRequest(row["region_id"], image.copy(), row["draft"])

        async def bounded() -> dict[str, Any]:
            async with semaphore:
                return await review_once(reviewer, request)

        answers = await asyncio.gather(*(bounded() for _ in range(samples)))
        record: dict[str, Any] = {k: row[k] for k in ("region_id", "source", "stratum", "series")}
        record["answers"] = answers
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        done += 1
        if done % 10 == 0 or done == len(rows):
            print(f"{done}/{len(rows)} {(time.monotonic() - started) / done:.1f} s/crop", flush=True)

    with (out / "results.jsonl").open("a") as sink:
        await asyncio.gather(*(one(r, sink) for r in rows))
    await reviewer.aclose()


def answers_of(record: dict[str, Any]) -> list[ReviewAnswer | None]:
    """The record's answers (None for a failed review); records of the first trial hold one answer inline."""
    stored = record.get("answers") or [record]
    return [
        ReviewAnswer(Verdict(a["verdict"]), tuple(Edit.from_json(e) for e in a["edits"]), a.get("model", ""))
        if "verdict" in a else None
        for a in stored
    ]  # fmt: skip


def only(answer: ReviewAnswer | None, ops: set[EditOp]) -> ReviewAnswer | None:
    """`answer` restricted to edits of the given kinds (the verdict becomes unchanged when none are left)."""
    if answer is None or answer.verdict is not Verdict.REVISED:
        return answer
    kept = tuple(e for e in answer.edits if e.op in ops)
    return dataclasses.replace(answer, edits=kept, verdict=Verdict.REVISED if kept else Verdict.UNCHANGED)


def policies(record: dict[str, Any]) -> dict[str, ReviewAnswer | None]:
    """Rules for using the stored answers, all scored on the same answers."""
    answers = answers_of(record)
    chosen: dict[str, ReviewAnswer | None] = {"single review": answers[0]}
    if len(answers) > 1:
        chosen["values need agreement"] = agree(answers) if all(a is not None for a in answers) else None  # type: ignore[arg-type]
    if any(e.op is EditOp.SET_CELL for a in answers if a is not None for e in a.edits):  # answers of earlier trials
        chosen["no value edits"] = only(answers[0], set(EditOp) - {EditOp.SET_CELL})
    chosen["header and title only"] = only(answers[0], {EditOp.REPLACE_HEADER, EditOp.SET_TITLE})
    return chosen


def first_passing(region_id: str, draft: str, answers: list[ReviewAnswer | None]) -> ReviewAnswer | None:
    """The first answer whose revision passes `judge`, else the first answer: several samples with the guards as
    the verifier."""
    for answer in answers:
        if answer is not None and judge(region_id, draft, answer).state is OutcomeState.REVISED:
            return answer
    return answers[0]


def terra_labels() -> list[dict[str, Any]]:
    """Every Terra label the references come from."""
    dirs = [Path(terra) for _, terra, _ in SOURCES.values()] + [
        d for d in EXTRA_TERRA if (d / "results.jsonl").exists()
    ]
    return [label for d in dirs for label in read_jsonl(d / "results.jsonl")]


def draft_files(crops: list[dict[str, Any]]) -> dict[str, Path]:
    """Crop id -> the prediction file its draft comes from, for crops lines that name one."""
    return {c["region_id"]: Path(c["drafts"]) for c in crops if c.get("drafts")}


def references_and_predictions(
    ids: set[str], run: Path = RUN, drafts: dict[str, Path] | None = None
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The reference tables (from the Terra labels) and the prediction records of these crops: the run's, or for
    crops in `drafts`, those of the file named there."""
    drafts = drafts or {}
    references = {
        label["region_id"]: crop_tables(reinterpret(label)) for label in terra_labels() if label["region_id"] in ids
    }
    predictions: dict[str, dict[str, Any]] = {}
    for _, (_, _, file) in SOURCES.items():
        for p in read_jsonl(run / file):
            if p["region_id"] in ids and p["region_id"] not in drafts:
                predictions[p["region_id"]] = p
    for file in set(drafts.values()):
        for p in read_jsonl(file):
            if drafts.get(p["region_id"]) == file and p["region_id"] in ids:
                predictions[p["region_id"]] = p
    return references, predictions


def drafted_rows(crops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Review rows for crops lines that name their drafts file (see the module docstring)."""
    manifest: dict[str, tuple[str, dict[str, Any]]] = {}
    for recrops, _, _ in SOURCES.values():
        for row in read_jsonl(Path(recrops) / "manifest.jsonl"):
            manifest[Path(row["image"]).stem] = (recrops, row)
    named = [c for c in crops if c.get("drafts")]
    _, predictions = references_and_predictions({c["region_id"] for c in named}, drafts=draft_files(named))
    rows: list[dict[str, Any]] = []
    for c in named:
        recrops, row = manifest[c["region_id"]]
        rows.append({"region_id": c["region_id"], "source": str(c["drafts"]), "stratum": c["stratum"],
                     "image": str(Path(recrops) / row["image"]), "series": row["source_id"],
                     "draft": predictions[c["region_id"]]["response"]["text"]})  # fmt: skip
    return rows


def tables_with(prediction: dict[str, Any], text: str | None = None) -> Any:
    """The prediction's tables, or those of `text` in its place (a revised draft)."""
    if text is not None:
        prediction = dict(prediction, response=dict(prediction["response"], text=text))
    return crop_tables(reinterpret(prediction))


def score(out: Path, run: Path, crops: set[str] | None = None, files: dict[str, Path] | None = None) -> None:
    results = {r["region_id"]: r for r in read_jsonl(out / "results.jsonl") if crops is None or r["region_id"] in crops}
    references, drafts = references_and_predictions(set(results), run, files)
    by: dict[str, dict[str, list[tuple[float, float, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for rid, r in results.items():
        draft = drafts[rid]["response"]["text"]
        before = crop_score(references[rid], tables_with(drafts[rid]))
        chosen = policies(r)
        answers = answers_of(r)
        for k in sorted({2, len(answers)} - {1}):
            if k <= len(answers):
                chosen[f"first passing revision of {k}"] = first_passing(rid, draft, answers[:k])
        for policy, answer in chosen.items():
            state, reason, after = "error", "", before
            if answer is not None:
                outcome = judge(rid, draft, answer)
                state, reason = outcome.state.value, outcome.reason
                if outcome.state.value == "revised":
                    after = crop_score(references[rid], tables_with(drafts[rid], outcome.text))
            by[policy][r["stratum"]].append((before, after, state, reason))
    for policy, strata in by.items():
        rng = random.Random(0)
        print(f"\n== {policy}")
        print(f"{'stratum':24s} {'crops':>5s} {'draft':>6s} {'review':>6s} {'diff [95%]':>22s}  better/worse  states")
        for stratum in (*STRATA, "all"):
            rows = [x for s in STRATA for x in strata[s]] if stratum == "all" else strata[stratum]
            if not rows:
                continue
            d = [a - b for b, a, _, _ in rows]
            boots = sorted(mean(rng.choices(d, k=len(d))) for _ in range(2000))
            states = Counter(state for _, _, state, _ in rows)
            better, worse = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
            print(
                f"{stratum:24s} {len(rows):5d} {mean(x[0] for x in rows):6.3f} {mean(x[1] for x in rows):6.3f} "
                f"{mean(d):+.3f} [{boots[50]:+.3f}, {boots[1949]:+.3f}]  {better:3d}/{worse:<3d}  {dict(states)}"
            )
        reasons = Counter(
            reason.split(":")[0] for x in strata.values() for *_, state, reason in x if state == "rejected"
        )
        print("rejections:", dict(reasons))
    stored = [a for r in results.values() for a in (r.get("answers") or [r])]
    usage: Counter[str] = Counter()
    for a in stored:
        usage.update(a.get("usage", {}))
    errors = Counter(a["error"].split(":")[0] for a in stored if "error" in a)
    providers = Counter(str((a.get("settings") or {}).get("provider")) for a in stored if "verdict" in a)
    print(f"providers: {dict(providers)}")
    cost = sum(a.get("cost") or 0.0 for a in stored)
    print(f"\n{len(stored)} reviews; errors: {dict(errors)}")
    print(f"usage {dict(usage)}; cost reported by OpenRouter ${cost:.2f}")


def retry_errors(out: Path) -> None:
    """Moves the records holding a failed answer from `results.jsonl` to `failed.jsonl`, so they are reviewed again."""
    records = read_jsonl(out / "results.jsonl")
    failed = [r for r in records if any("error" in a for a in r["answers"])]
    with (out / "failed.jsonl").open("a") as sink:
        sink.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in failed)
    kept = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records if r not in failed)
    (out / "results.jsonl").write_text(kept)
    print(f"{len(failed)} crops with a failed answer moved to failed.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=Path("data/reviews/haiku-shift"))
    parser.add_argument("--samples", type=int, default=1, help="independent reviews per crop")
    parser.add_argument("--per-stratum", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--model", default="anthropic/claude-haiku-5.5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--quantization", action="append", default=[], help="only providers at this precision")
    parser.add_argument("--provider-sort", choices=("price", "throughput", "latency"), help="OpenRouter routing order")
    parser.add_argument("--crops", type=Path, help="review or score these crops (JSON lines with region_id)")
    parser.add_argument("--url", default=OPENROUTER_URL, help="chat completions URL, e.g. a local vLLM server's")
    parser.add_argument(
        "--template-kwargs", type=json.loads, default=None, help='chat template options, e.g. {"enable_thinking": true}'
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--assumed-output-tokens", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true", help="sample and estimate the cost; no API calls")
    parser.add_argument("--score", action="store_true", help="score the results in --out")
    parser.add_argument(
        "--retry-errors", action="store_true", help="review crops with a failed answer again (moved to failed.jsonl)"
    )
    args = parser.parse_args()
    lines = read_jsonl(args.crops) if args.crops else []
    chosen = {r["region_id"] for r in lines} if args.crops else None
    files = draft_files(lines)
    if args.score:
        score(args.out, args.run, chosen, files)
        return
    if chosen is None:
        rows = sample(candidates(args.run), args.per_stratum, args.seed)
    else:
        rows = [r for r in candidates(args.run) if r["region_id"] in chosen and r["region_id"] not in files]
        rows += drafted_rows(lines)
        if missing := chosen - {r["region_id"] for r in rows}:
            raise SystemExit(f"{len(missing)} crops of {args.crops} are not reviewable validation crops")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sample.jsonl").write_text("".join(json.dumps({"region_id": r["region_id"]}) + "\n" for r in rows))
    print(Counter(r["stratum"] for r in rows))
    if args.dry_run:
        reviewer = OpenRouterReviewer(
            model=args.model, effort=args.effort, api_key="dry-run", max_tokens=args.max_tokens
        )
        estimate(rows, reviewer, args.assumed_output_tokens, args.samples)
        return
    if args.retry_errors:
        retry_errors(args.out)
    done = (
        {r["region_id"] for r in read_jsonl(args.out / "results.jsonl")}
        if (args.out / "results.jsonl").exists()
        else set()
    )
    todo = [r for r in rows if r["region_id"] not in done]
    print(f"{len(todo)} crops to review ({len(done)} done)", flush=True)
    reviewer = OpenRouterReviewer(
        model=args.model, effort=args.effort, max_tokens=args.max_tokens, quantizations=args.quantization,
        url=args.url, chat_template_kwargs=args.template_kwargs, provider_sort=args.provider_sort,
    )  # fmt: skip
    asyncio.run(review_all(todo, reviewer, args.out, args.concurrency, args.samples))
    score(args.out, args.run, chosen, files)


if __name__ == "__main__":
    main()
