"""Interpreting responses and running recognizers: outcomes, ordering, concurrency, retries."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from document_prep.models import Box, DocumentSource, PreparationSettings, PreparedRegion, TableRegion
from extraction.extract import ExtractionIssue, extract_regions, interpret
from extraction.results import (
    ComponentFailure,
    FinishReason,
    OcrResult,
    RegionResult,
    RegionStatus,
    encode_region_result,
)
from extraction.saved import SavedResponses
from table_core.codec import schema_errors
from table_core.issues import CoreIssue, TableOutcome

EXAMPLE = """# Receipts (in dollars)
<table>
  <thead><tr><th>Item</th><th>2025</th></tr></thead>
  <tbody>
    <tr><td data-level="0">Receipts:</td><td></td></tr>
    <tr><td data-level="1">Taxes</td><td>100<sup>1</sup></td></tr>
  </tbody>
  <tfoot><tr><td colspan="2">1/ Provisional figure.</td></tr></tfoot>
</table>"""
BOX = Box(left=0, top=0, right=10, bottom=10)


def region(region_id: str) -> PreparedRegion:
    return PreparedRegion(
        source=DocumentSource(path=Path("x.pdf"), sha256="0" * 64),
        region=TableRegion(region_id=region_id, page=1, box=BOX),
        page_size=(10.0, 10.0),
        page_pixels=(10, 10),
        detected_px=BOX,
        crop_px=BOX,
        crop_box=BOX,
        image=Image.new("RGB", (10, 10), "white"),
        settings=PreparationSettings(),
        renderer="test",
    )


def response(text: str, finish: FinishReason = FinishReason.STOP) -> OcrResult:
    return OcrResult(text=text, model="test", finish_reason=finish)


def codes(result: RegionResult) -> list[str]:
    return [i.code for i in result.issues]


# Interpreting one response


def test_ready_table() -> None:
    result = interpret("r", response(EXAMPLE))
    assert result.status is RegionStatus.TABLES
    (table,) = result.tables
    assert (table.table_id, table.index, table.outcome) == ("r/t0", 0, TableOutcome.READY)
    assert table.publishable is table.candidate
    assert table.candidate is not None
    assert table.candidate.id == "r/t0"
    assert table.sources["r1c1"].row == 1


def test_not_a_table() -> None:
    result = interpret("r", response("NOT_A_TABLE"))
    assert (result.status, result.tables, result.error) == (RegionStatus.NOT_A_TABLE, (), None)


@pytest.mark.parametrize("text", ["", "Sorry, I cannot read this image."])
def test_unusable_response_fails_the_region(text: str) -> None:
    result = interpret("r", response(text))
    assert result.status is RegionStatus.FAILED
    assert result.error == "the response holds no usable table"
    assert result.response is not None


def test_truncated_response_keeps_candidates_but_none_is_ready() -> None:
    result = interpret("r", response(EXAMPLE, FinishReason.LENGTH))
    assert codes(result) == [ExtractionIssue.TRUNCATED]
    (table,) = result.tables
    assert table.outcome is TableOutcome.UNRESOLVED
    assert table.candidate is not None
    assert ExtractionIssue.TRUNCATED in [i.code for i in table.issues]


def test_truncated_sentinel_is_not_a_non_table() -> None:
    result = interpret("r", response("NOT_A_TABLE", FinishReason.LENGTH))
    assert result.status is RegionStatus.FAILED


def test_each_fragment_has_its_own_outcome() -> None:
    bad = "<table><tbody><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></tbody></table>"  # ragged row
    result = interpret("r", response(EXAMPLE + "\n" + bad))
    assert [t.outcome for t in result.tables] == [TableOutcome.READY, TableOutcome.UNRESOLVED]
    assert CoreIssue.UNCOVERED_POSITION in [i.code for i in result.tables[1].issues]
    assert [t.table_id for t in result.tables] == ["r/t0", "r/t1"]


def test_response_level_issues_reach_every_table() -> None:
    result = interpret("r", response("Here is the table:\n" + EXAMPLE))
    assert codes(result) == [CoreIssue.UNEXPECTED_CONTENT]
    assert result.tables[0].outcome is TableOutcome.UNRESOLVED


def test_encoded_result_is_json_with_schema_valid_candidates() -> None:
    payload = encode_region_result(interpret("r", response(EXAMPLE, FinishReason.LENGTH)))
    text = json.dumps(payload)
    decoded = json.loads(text)
    assert decoded["status"] == "tables"
    assert decoded["response"]["finish_reason"] == "length"
    assert decoded["issues"][0]["code"] == "truncated"
    candidate = decoded["tables"][0]["candidate"]
    assert schema_errors(candidate) == ()


# Running a recognizer over many regions


class Fake:
    """Answers after a per-region delay, tracking how many calls run at once; can fail the first attempts."""

    def __init__(self, delays: dict[str, float], failures: dict[str, list[ComponentFailure]] | None = None) -> None:
        self.delays = delays
        self.failures = failures or {}
        self.active = 0
        self.peak = 0
        self.calls: list[str] = []

    async def recognize(self, region: PreparedRegion) -> OcrResult:
        region_id = region.region.region_id
        self.calls.append(region_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delays.get(region_id, 0))
            pending = self.failures.get(region_id)
            if pending:
                raise pending.pop(0)
            return response(EXAMPLE)
        finally:
            self.active -= 1


def collect(regions: list[PreparedRegion], recognizer: Fake, **kwargs: int) -> list[RegionResult]:
    async def run() -> list[RegionResult]:
        return [r async for r in extract_regions(regions, recognizer, retry_delay=0, **kwargs)]

    return asyncio.run(run())


def test_results_come_in_source_order_with_bounded_concurrency() -> None:
    ids = [f"r{i}" for i in range(10)]
    fake = Fake({rid: 0.05 * (10 - i) / 10 for i, rid in enumerate(ids)})  # later regions finish first
    results = collect([region(i) for i in ids], fake, max_in_flight=3)
    assert [r.region_id for r in results] == ids
    assert fake.peak == 3


def test_retryable_failures_are_retried() -> None:
    fake = Fake({}, {"a": [ComponentFailure("timeout", "slow", retryable=True)]})
    (result,) = collect([region("a")], fake)
    assert (result.status, result.attempts) == (RegionStatus.TABLES, 2)


def test_retries_are_bounded() -> None:
    failures = [ComponentFailure("codex_failed", "rate limited", retryable=True) for _ in range(5)]
    fake = Fake({}, {"a": failures})
    (result,) = collect([region("a")], fake, max_attempts=3)
    assert (result.status, result.attempts) == (RegionStatus.FAILED, 3)
    assert result.error == "codex_failed: rate limited"


def test_other_regions_survive_a_failure() -> None:
    fake = Fake({}, {"b": [ComponentFailure("broken", "bad", retryable=False)]})
    results = collect([region("a"), region("b"), region("c")], fake)
    assert [(r.region_id, r.status, r.attempts) for r in results] == [
        ("a", RegionStatus.TABLES, 1),
        ("b", RegionStatus.FAILED, 1),
        ("c", RegionStatus.TABLES, 1),
    ]


def test_regions_are_pulled_lazily() -> None:
    pulled: list[int] = []

    def lazy() -> Iterator[PreparedRegion]:
        for i in range(100):
            pulled.append(i)
            yield region(f"r{i}")

    async def first() -> RegionResult:
        stream = extract_regions(lazy(), Fake({}), max_in_flight=2, retry_delay=0)
        result = await anext(stream)
        await stream.aclose()  # type: ignore[attr-defined]
        return result

    assert asyncio.run(first()).region_id == "r0"
    assert len(pulled) <= 4 + 1


def test_stopping_early_cancels_calls_in_flight() -> None:
    fake = Fake({"slow": 10.0})

    async def run() -> None:
        stream = extract_regions([region("fast"), region("slow")], fake, max_in_flight=2, retry_delay=0)
        assert (await anext(stream)).region_id == "fast"
        await stream.aclose()  # type: ignore[attr-defined]

    asyncio.run(asyncio.wait_for(run(), 2))
    assert fake.active == 0


def test_saved_responses() -> None:
    saved = SavedResponses({"a": response("NOT_A_TABLE")})
    results = collect([region("a"), region("b")], saved)  # type: ignore[arg-type]
    assert [r.status for r in results] == [RegionStatus.NOT_A_TABLE, RegionStatus.FAILED]
    assert results[1].error == "no_saved_response: b"


def test_max_in_flight_must_be_positive() -> None:
    with pytest.raises(ValueError):
        collect([region("a")], Fake({}), max_in_flight=0)
