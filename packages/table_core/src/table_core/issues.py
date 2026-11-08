"""Issues, source references and table outcomes.

Expected content problems are `Issue` records; operational failures raise. `Issue` is shared by every module
(preparation, recognition and review add their own codes), so `code` is a plain string. The codes table_core
emits are listed in `CoreIssue`, each with a fixed default for whether it blocks publication.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from table_core.models import Table


class Stage(StrEnum):
    """The pipeline stage that raised an issue."""

    PREPARATION = "preparation"
    RECOGNITION = "recognition"
    PARSE = "parse"
    STRUCTURE = "structure"
    VALIDATION = "validation"
    REVIEW = "review"


class Section(StrEnum):
    """The part of a transcribed table a source element belongs to."""

    TITLE = "title"
    HEAD = "head"
    BODY = "body"
    FOOT = "foot"


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceRef:
    """A position in the raw transcription: fragment (table) index, section, and `tr`/cell indexes within it.

    Indexes count elements as written, before span expansion. `row` and `cell` are None for a whole fragment
    or section; the title has neither.
    """

    fragment: int
    section: Section | None = None
    row: int | None = None
    cell: int | None = None

    def __post_init__(self) -> None:
        if any(i is not None and i < 0 for i in (self.fragment, self.row, self.cell)):
            raise ValueError("source indexes must be nonnegative")
        if self.row is not None and self.section in (None, Section.TITLE):
            raise ValueError("a row index needs a table section (head, body or foot)")
        if self.cell is not None and self.row is None:
            raise ValueError("a cell index needs a row index")


@dataclass(frozen=True, slots=True, kw_only=True)
class Issue:
    """One problem found in a response, a table, or a stage's processing.

    `object_id` points at a table object (header, column, row, cell, note) when one exists; `source` points at
    the transcription. Response-level issues have neither. The table an issue belongs to is given by where the
    issue is stored (a table result or a region result), not by a field here.
    """

    code: str
    message: str
    stage: Stage
    blocking: bool
    object_id: str | None = None
    source: SourceRef | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("issue code must be nonempty")
        if self.object_id == "":
            raise ValueError("issue object_id must be nonempty when present")


class CoreIssue(StrEnum):
    """Issue codes raised by table_core. Codes listed in `NON_BLOCKING` do not block publication; all others do."""

    # Response level (parse_output)
    EMPTY_RESPONSE = "empty_response"
    MIXED_SENTINEL = "mixed_sentinel_and_tables"
    PIPE_TABLE = "pipe_table"
    CAPTION_ELEMENT = "caption_element"
    UNEXPECTED_CONTENT = "unexpected_content"  # prose, code fences, stray text between fragments
    # Fragment level (parse_output)
    MALFORMED_FRAGMENT = "malformed_fragment"  # no usable table structure could be read
    UNTERMINATED_TABLE = "unterminated_table"  # <table> never closed; the fragment runs to the end of the response
    NESTED_TABLE = "nested_table"
    HTML_REPAIRED = "html_repaired"  # the HTML parser had to repair the markup
    UNEXPECTED_MARKUP = "unexpected_markup"  # elements or attributes outside the contract, e.g. <br>, style
    INVALID_SPAN = "invalid_span"  # rowspan/colspan that is not a positive integer
    INVALID_LEVEL = "invalid_level"  # data-level that is not a nonnegative integer; level is omitted
    TITLE_FORMAT = "title_format"  # heading level other than 1, or several heading lines before one table
    # Proposal (propose_table)
    MISSING_LEVEL = "missing_level"
    LABEL_SPANS_VALUES = "label_spans_values"
    MALFORMED_FOOTNOTE_ROW = "malformed_footnote_row"  # tfoot row that is not one full-width cell
    UNRESOLVED_MARKER = "unresolved_marker"  # no note with a matching label
    AMBIGUOUS_NOTE_LABEL = "ambiguous_note_label"  # several notes match a marker
    # Validation (validate_table)
    DUPLICATE_ID = "duplicate_id"
    DANGLING_REFERENCE = "dangling_reference"
    UNREFERENCED_HEADER = "unreferenced_header"
    HEADER_STRUCTURE = "header_structure"  # columns sharing a header are not adjacent
    NONCONSECUTIVE_COLUMNS = "nonconsecutive_columns"
    CELL_ORDER = "cell_order"
    UNCOVERED_POSITION = "uncovered_position"
    OVERLAPPING_CELLS = "overlapping_cells"
    ROWSPAN_OUT_OF_BOUNDS = "rowspan_out_of_bounds"
    MISPLACED_LEVEL = "misplaced_level"
    LABEL_COLUMNS_OUT_OF_RANGE = "label_columns_out_of_range"
    INVALID_MARKER_POSITION = "invalid_marker_position"
    DUPLICATE_NOTE_REF = "duplicate_note_ref"
    INVALID_UNICODE = "invalid_unicode"


NON_BLOCKING: frozenset[str] = frozenset(
    {
        CoreIssue.HTML_REPAIRED,
        CoreIssue.UNEXPECTED_MARKUP,
        CoreIssue.INVALID_LEVEL,
        CoreIssue.TITLE_FORMAT,
        CoreIssue.MISSING_LEVEL,
        CoreIssue.LABEL_SPANS_VALUES,
        CoreIssue.MALFORMED_FOOTNOTE_ROW,
        CoreIssue.UNRESOLVED_MARKER,
        CoreIssue.AMBIGUOUS_NOTE_LABEL,
    }
)
"""Core codes that do not block publication. Every other core code blocks."""


def core_issue(
    code: CoreIssue,
    message: str,
    stage: Stage,
    *,
    object_id: str | None = None,
    source: SourceRef | None = None,
) -> Issue:
    """An `Issue` for a core code, with `blocking` taken from the code's default."""
    return Issue(
        code=code.value,
        message=message,
        stage=stage,
        blocking=code not in NON_BLOCKING,
        object_id=object_id,
        source=source,
    )


class TableOutcome(StrEnum):
    """Outcome of one extracted table (architecture: outcomes and failure policy)."""

    READY = "ready"  # structurally valid, no blocking issues; not independently verified
    UNRESOLVED = "unresolved"  # a candidate exists but blocking issues prevent publication
    FAILED = "failed"  # no candidate could be constructed


def table_outcome(candidate: Table | None, issues: Iterable[Issue]) -> TableOutcome:
    """FAILED without a candidate, UNRESOLVED if any issue blocks, READY otherwise.

    Callers add response-level blocking issues (contract conflicts, truncation) to `issues`, so they make
    every affected table non-ready.
    """
    if candidate is None:
        return TableOutcome.FAILED
    if any(issue.blocking for issue in issues):
        return TableOutcome.UNRESOLVED
    return TableOutcome.READY
