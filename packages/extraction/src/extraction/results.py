"""Recognition responses and extraction results (planning/architecture.md, "Outcomes and failure policy").

A region ends as `tables`, `not_a_table` or `failed`. Only a completed, non-truncated response that is exactly
the sentinel gives `not_a_table`; empty or unsupported output and operational failures give `failed`. A `tables`
region holds one `TableResult` per table fragment, each `ready`, `unresolved` or `failed`, so one bad fragment
does not sink its neighbours.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from table_core.codec import JsonObject, encode_table
from table_core.issues import Issue, SourceRef, TableOutcome
from table_core.models import Table


class FinishReason(StrEnum):
    STOP = "stop"  # the model finished its answer
    LENGTH = "length"  # the answer was cut off at a token limit


@dataclass(frozen=True, slots=True, kw_only=True)
class OcrResult:
    """A recognizer's raw response for one region, and how it was produced.

    `model` names the model (release) and `settings` the generation settings actually used; `usage` holds
    whatever token or cost accounting the recognizer reports.
    """

    text: str
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    settings: Mapping[str, Any] = field(default_factory=dict[str, Any])
    usage: Mapping[str, int] = field(default_factory=dict[str, int])


class ComponentFailure(Exception):
    """An operational failure (timeout, crashed tool, rate limit), as opposed to bad content.

    `retryable` failures are retried a bounded number of times; the rest fail the region at once.
    """

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


class RegionStatus(StrEnum):
    TABLES = "tables"
    NOT_A_TABLE = "not_a_table"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class TableResult:
    """One table fragment of a region: its candidate (or None), outcome and issues.

    `table_id` is `<region_id>/t<index>`, the index counting table fragments in reading order.
    """

    table_id: str
    region_id: str
    index: int
    outcome: TableOutcome
    candidate: Table | None
    issues: tuple[Issue, ...] = ()
    sources: Mapping[str, SourceRef] = field(default_factory=dict[str, SourceRef])

    @property
    def publishable(self) -> Table | None:
        return self.candidate if self.outcome is TableOutcome.READY else None


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionResult:
    """Everything extraction found for one region, in source order with the others.

    `issues` are response-level (for example truncation or text outside tables) and also appear on each table
    they affect. `error` explains a failed region; `attempts` counts recognition calls.
    """

    region_id: str
    status: RegionStatus
    tables: tuple[TableResult, ...] = ()
    response: OcrResult | None = None
    issues: tuple[Issue, ...] = ()
    error: str | None = None
    attempts: int = 1


def encode_issue(issue: Issue) -> JsonObject:
    payload: JsonObject = {
        "code": issue.code,
        "message": issue.message,
        "stage": issue.stage.value,
        "blocking": issue.blocking,
    }
    if issue.object_id is not None:
        payload["object_id"] = issue.object_id
    if issue.source is not None:
        source: JsonObject = {"fragment": issue.source.fragment}
        if issue.source.section is not None:
            source["section"] = issue.source.section.value
        if issue.source.row is not None:
            source["row"] = issue.source.row
        if issue.source.cell is not None:
            source["cell"] = issue.source.cell
        payload["source"] = source
    return payload


def encode_region_result(result: RegionResult) -> JsonObject:
    """A JSON object for saving a region result, including the raw response and every candidate."""
    payload: JsonObject = {"region_id": result.region_id, "status": result.status.value, "attempts": result.attempts}
    if result.error is not None:
        payload["error"] = result.error
    if result.response is not None:
        r = result.response
        payload["response"] = {
            "text": r.text,
            "model": r.model,
            "finish_reason": r.finish_reason.value,
            "settings": dict(r.settings),
            "usage": dict(r.usage),
        }
    payload["issues"] = [encode_issue(i) for i in result.issues]
    payload["tables"] = [
        {
            "table_id": t.table_id,
            "index": t.index,
            "outcome": t.outcome.value,
            "candidate": encode_table(t.candidate) if t.candidate is not None else None,
            "issues": [encode_issue(i) for i in t.issues],
        }
        for t in result.tables
    ]
    return payload
