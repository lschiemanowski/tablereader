"""Table JSON model, draft 0.3 (docs/table-format.md).

One frozen dataclass per JSON object. Fields mirror the JSON field names; sequences are tuples so a `Table` is
immutable and hashable. Optional JSON fields have defaults equal to the consumer defaults in the spec, and the
encoder omits a field whose value equals its default.

Constructors enforce local shape only, the same rules the JSON Schema states for a single object: nonempty
IDs, `rowspan >= 1`, `level >= 0`, `at >= 0`, marker text with a non-whitespace character, `label_columns >= 1`,
at least one column, and no repeated ID within one ID list. They raise `ValueError`. Everything that relates
one object to another (references, grid coverage, header intervals, marker positions relative to text) is
checked by `validation.validate_table`, so a `Table` can hold a structurally invalid candidate for inspection.

IDs are plain strings and opaque, as in the spec.
"""

from collections.abc import Sequence
from dataclasses import dataclass


def _check_id(value: str, what: str) -> None:
    if not value:
        raise ValueError(f"{what} must be a nonempty string")


def _check_ids(values: Sequence[str], what: str) -> None:
    for value in values:
        _check_id(value, f"each entry of {what}")
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must not repeat an ID")


@dataclass(frozen=True, slots=True, kw_only=True)
class Marker:
    """A footnote marker removed from the containing object's text.

    `at` is the code-point position in that text where the marker was printed (0..len(text)).
    `note_ref` is the uniquely linked note, or None when unresolved.
    """

    at: int
    text: str
    note_ref: str | None = None

    def __post_init__(self) -> None:
        if self.at < 0:
            raise ValueError("marker position must be nonnegative")
        if not self.text or self.text.isspace():
            raise ValueError("marker text must contain a non-whitespace character")
        if self.note_ref is not None:
            _check_id(self.note_ref, "marker note_ref")


@dataclass(frozen=True, slots=True, kw_only=True)
class TextBlock:
    """A table title: text plus its markers, note attachments and uncertainty flag."""

    text: str
    markers: tuple[Marker, ...] = ()
    note_refs: tuple[str, ...] = ()
    unclear: bool = False

    def __post_init__(self) -> None:
        _check_ids(self.note_refs, "title note_refs")


@dataclass(frozen=True, slots=True, kw_only=True)
class Header:
    """One header cell, stored once and shared by the paths of every column below it."""

    id: str
    text: str
    markers: tuple[Marker, ...] = ()
    note_refs: tuple[str, ...] = ()
    unclear: bool = False

    def __post_init__(self) -> None:
        _check_id(self.id, "header id")
        _check_ids(self.note_refs, "header note_refs")


@dataclass(frozen=True, slots=True, kw_only=True)
class Column:
    """A leaf column: its header IDs from outermost to innermost (may be empty)."""

    id: str
    headers: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_id(self.id, "column id")
        _check_ids(self.headers, "column headers")


@dataclass(frozen=True, slots=True, kw_only=True)
class Cell:
    """A body cell, stored once in its starting row.

    `columns` are the consecutive leaf columns it covers; it covers `rowspan` rows starting at its row.
    `level` is None when no indentation was recorded, which is not the same as level 0.
    """

    id: str
    columns: tuple[str, ...]
    text: str
    rowspan: int = 1
    level: int | None = None
    unclear: bool = False
    markers: tuple[Marker, ...] = ()
    note_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_id(self.id, "cell id")
        if not self.columns:
            raise ValueError("cell must cover at least one column")
        _check_ids(self.columns, "cell columns")
        if self.rowspan < 1:
            raise ValueError("rowspan must be at least 1")
        if self.level is not None and self.level < 0:
            raise ValueError("level must be nonnegative")
        _check_ids(self.note_refs, "cell note_refs")


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """A body row: its ID and the cells that start in it, ordered by first covered column."""

    id: str
    cells: tuple[Cell, ...]

    def __post_init__(self) -> None:
        _check_id(self.id, "row id")


@dataclass(frozen=True, slots=True, kw_only=True)
class Note:
    """A footnote line. `label` is the printed label (`1/`, `*`) when one was separated from the text."""

    id: str
    text: str
    label: str | None = None
    unclear: bool = False

    def __post_init__(self) -> None:
        _check_id(self.id, "note id")
        if self.label is not None and not self.label:
            raise ValueError("note label must be nonempty when present")


@dataclass(frozen=True, slots=True, kw_only=True)
class Table:
    """One logical table. `schema_version` is not a field: the codec writes and checks it."""

    id: str
    headers: tuple[Header, ...]
    columns: tuple[Column, ...]
    rows: tuple[Row, ...]
    title: TextBlock | None = None
    label_columns: int = 1
    notes: tuple[Note, ...] = ()
    note_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_id(self.id, "table id")
        if not self.columns:
            raise ValueError("table must have at least one column")
        if self.label_columns < 1:
            raise ValueError("label_columns must be at least 1")
        _check_ids(self.note_refs, "table note_refs")


def with_markers(text: str, markers: Sequence[Marker]) -> str:
    """The transcribed string with marker text reinserted at each marker's position.

    `with_markers("10", (Marker(at=2, text="1"),)) == "101"`. Markers at the same position are inserted in
    array order. Used for display and for comparing with legacy representations, never for reading values.
    Raises `ValueError` if a position lies outside the text or positions decrease.
    """
    pieces: list[str] = []
    pos = 0
    for marker in markers:
        if marker.at < pos or marker.at > len(text):
            raise ValueError(f"marker position {marker.at} is out of order or outside the text")
        pieces.append(text[pos : marker.at])
        pieces.append(marker.text)
        pos = marker.at
    pieces.append(text[pos:])
    return "".join(pieces)
