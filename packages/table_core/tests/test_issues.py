"""Issues, source references and table outcomes."""

from collections.abc import Callable

import pytest

from table_core.issues import (
    NON_BLOCKING,
    CoreIssue,
    Issue,
    Section,
    SourceRef,
    Stage,
    TableOutcome,
    core_issue,
    table_outcome,
)
from table_core.models import Column, Table

TABLE = Table(id="t", headers=(), columns=(Column(id="c0", headers=()),), rows=())


def test_core_codes_are_distinct() -> None:
    # A repeated value would silently turn one member into an alias of another.
    assert len(CoreIssue.__members__) == len({m.value for m in CoreIssue.__members__.values()})


def test_non_blocking_codes_are_core_codes() -> None:
    assert set(NON_BLOCKING) <= set(CoreIssue)


@pytest.mark.parametrize("code", list(CoreIssue), ids=lambda c: c.value)
def test_core_issue_takes_blocking_from_code(code: CoreIssue) -> None:
    issue = core_issue(code, "message", Stage.PARSE)
    assert issue.blocking == (code not in NON_BLOCKING)
    assert issue.code == code.value
    assert type(issue.code) is str


def test_agreed_non_blocking_set() -> None:
    assert {CoreIssue(c).value for c in NON_BLOCKING} == {
        "html_repaired",
        "unexpected_markup",
        "invalid_level",
        "title_format",
        "missing_level",
        "label_spans_values",
        "malformed_footnote_row",
        "unresolved_marker",
        "ambiguous_note_label",
    }


def test_core_issue_keeps_references() -> None:
    source = SourceRef(fragment=1, section=Section.BODY, row=3, cell=0)
    issue = core_issue(CoreIssue.MISSING_LEVEL, "no data-level", Stage.STRUCTURE, object_id="r3c0", source=source)
    assert (issue.object_id, issue.source) == ("r3c0", source)


def test_other_modules_can_use_their_own_codes() -> None:
    issue = Issue(code="truncated_response", message="hit token cap", stage=Stage.RECOGNITION, blocking=True)
    assert table_outcome(TABLE, [issue]) is TableOutcome.UNRESOLVED


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: SourceRef(fragment=-1), id="negative fragment"),
        pytest.param(lambda: SourceRef(fragment=0, section=Section.BODY, row=-1), id="negative row"),
        pytest.param(lambda: SourceRef(fragment=0, section=Section.BODY, row=0, cell=-1), id="negative cell"),
        pytest.param(lambda: SourceRef(fragment=0, row=0), id="row without section"),
        pytest.param(lambda: SourceRef(fragment=0, section=Section.TITLE, row=0), id="row in title"),
        pytest.param(lambda: SourceRef(fragment=0, section=Section.BODY, cell=0), id="cell without row"),
        pytest.param(lambda: Issue(code="", message="m", stage=Stage.PARSE, blocking=True), id="empty code"),
        pytest.param(
            lambda: Issue(code="c", message="m", stage=Stage.PARSE, blocking=True, object_id=""), id="empty object_id"
        ),
    ],
)
def test_bad_references_are_rejected(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()


def test_valid_source_refs() -> None:
    SourceRef(fragment=0)
    SourceRef(fragment=0, section=Section.TITLE)
    SourceRef(fragment=2, section=Section.FOOT, row=0)


def test_outcome_without_candidate_is_failed() -> None:
    assert table_outcome(None, []) is TableOutcome.FAILED


def test_outcome_with_blocking_issue_is_unresolved() -> None:
    issues = [
        core_issue(CoreIssue.MISSING_LEVEL, "m", Stage.STRUCTURE),
        core_issue(CoreIssue.UNCOVERED_POSITION, "m", Stage.VALIDATION),
    ]
    assert table_outcome(TABLE, issues) is TableOutcome.UNRESOLVED


def test_outcome_with_only_non_blocking_issues_is_ready() -> None:
    issues = [core_issue(CoreIssue.MISSING_LEVEL, "m", Stage.STRUCTURE)]
    assert table_outcome(TABLE, issues) is TableOutcome.READY
    assert table_outcome(TABLE, []) is TableOutcome.READY


def test_outcome_accepts_any_iterable() -> None:
    issues = (i for i in [core_issue(CoreIssue.PIPE_TABLE, "m", Stage.PARSE)])
    assert table_outcome(TABLE, issues) is TableOutcome.UNRESOLVED
