"""Turn recognizer responses into checked table results, with bounded concurrency and results in source order."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable
from enum import StrEnum
from typing import Protocol

from document_prep.models import PreparedRegion
from extraction.results import ComponentFailure, FinishReason, OcrResult, RegionResult, RegionStatus, TableResult
from table_core.issues import Issue, Stage, table_outcome
from table_core.parsing import ResponseKind, parse_output
from table_core.structure import propose_table
from table_core.validation import validate_table


class TableRecognizer(Protocol):
    """Transcribes one prepared region into the fixed OCR transcription format.

    Raises `ComponentFailure` for operational problems; bad content is returned as text and judged by parsing.
    """

    async def recognize(self, region: PreparedRegion) -> OcrResult: ...


class ExtractionIssue(StrEnum):
    """Issue codes added by extraction (table_core's codes are `CoreIssue`)."""

    TRUNCATED = "truncated"  # the response was cut off; nothing in it is ready


def interpret(region_id: str, response: OcrResult, *, attempts: int = 1) -> RegionResult:
    """Parse, propose and validate one response. Deterministic; never raises on bad content."""
    output = parse_output(response.text)
    issues = list(output.issues)
    truncated = response.finish_reason is FinishReason.LENGTH
    if truncated:
        issues.append(
            Issue(
                code=ExtractionIssue.TRUNCATED.value,
                message="the response was cut off at the token limit",
                stage=Stage.RECOGNITION,
                blocking=True,
            )
        )
    response_issues = tuple(issues)
    if output.kind is ResponseKind.NOT_A_TABLE and not truncated:
        return RegionResult(region_id=region_id, status=RegionStatus.NOT_A_TABLE, response=response, attempts=attempts)
    if output.kind is not ResponseKind.TABLES:
        return RegionResult(
            region_id=region_id,
            status=RegionStatus.FAILED,
            response=response,
            issues=response_issues,
            error="the response holds no usable table" if output.kind is ResponseKind.INVALID else "truncated",
            attempts=attempts,
        )
    tables: list[TableResult] = []
    for index, fragment in enumerate(output.fragments):
        table_id = f"{region_id}/t{index}"
        found = response_issues + fragment.issues
        if fragment.table is None:
            tables.append(_failed(table_id, region_id, index, found))
            continue
        proposal = propose_table(fragment.table, table_id=table_id)
        found += proposal.issues
        if proposal.table is None:
            tables.append(_failed(table_id, region_id, index, found))
            continue
        found += validate_table(proposal.table)
        tables.append(
            TableResult(
                table_id=table_id,
                region_id=region_id,
                index=index,
                outcome=table_outcome(proposal.table, found),
                candidate=proposal.table,
                issues=found,
                sources=proposal.sources,
            )
        )
    return RegionResult(
        region_id=region_id,
        status=RegionStatus.TABLES,
        tables=tuple(tables),
        response=response,
        issues=response_issues,
        attempts=attempts,
    )


def _failed(table_id: str, region_id: str, index: int, issues: tuple[Issue, ...]) -> TableResult:
    return TableResult(
        table_id=table_id, region_id=region_id, index=index, outcome=table_outcome(None, issues), candidate=None,
        issues=issues,
    )  # fmt: skip


async def recognize_region(
    region: PreparedRegion, recognizer: TableRecognizer, *, max_attempts: int = 3, retry_delay: float = 5.0
) -> RegionResult:
    """Recognize and interpret one region, retrying retryable operational failures with growing delays."""
    region_id = region.region.region_id
    for attempt in range(1, max_attempts + 1):
        try:
            response = await recognizer.recognize(region)
        except ComponentFailure as failure:
            if failure.retryable and attempt < max_attempts:
                await asyncio.sleep(retry_delay * 2 ** (attempt - 1))
                continue
            return RegionResult(region_id=region_id, status=RegionStatus.FAILED, error=str(failure), attempts=attempt)
        return interpret(region_id, response, attempts=attempt)
    raise AssertionError("unreachable")


async def extract_regions(
    regions: Iterable[PreparedRegion],
    recognizer: TableRecognizer,
    *,
    max_in_flight: int = 4,
    max_attempts: int = 3,
    retry_delay: float = 5.0,
) -> AsyncIterator[RegionResult]:
    """Results for `regions`, in their order, with at most `max_in_flight` recognition calls at a time.

    Regions are pulled from the iterable only as capacity frees up, so a lazy iterable keeps at most
    `2 * max_in_flight` prepared images in memory. Cancelling the iteration cancels the calls in flight.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")
    gate = asyncio.Semaphore(max_in_flight)

    async def run(region: PreparedRegion) -> RegionResult:
        async with gate:
            return await recognize_region(region, recognizer, max_attempts=max_attempts, retry_delay=retry_delay)

    pending: deque[asyncio.Task[RegionResult]] = deque()
    source = iter(regions)
    try:
        exhausted = False
        while True:
            while not exhausted and len(pending) < 2 * max_in_flight:
                region = next(source, None)
                if region is None:
                    exhausted = True
                else:
                    pending.append(asyncio.create_task(run(region)))
            if not pending:
                return
            yield await pending.popleft()
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
