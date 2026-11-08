"""Structural validation of tables."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from table_core.codec import decode_table
from table_core.issues import CoreIssue, Issue, Stage
from table_core.models import Cell, Column, Header, Marker, Note, Row, Table, TextBlock
from table_core.parsing import parse_output
from table_core.structure import propose_table
from table_core.validation import validate_table

EXAMPLES = Path(__file__).parents[3] / "examples"


def base(**changes: object) -> Table:
    """A valid 2x2 table: a stub column and one value column, one note linked from a value."""
    table = Table(
        id="t",
        title=TextBlock(text="Receipts"),
        headers=(Header(id="h0", text="Item"), Header(id="h1", text="2025")),
        columns=(Column(id="c0", headers=("h0",)), Column(id="c1", headers=("h1",))),
        rows=(
            Row(
                id="r0",
                cells=(Cell(id="a", columns=("c0",), text="Taxes", level=0), Cell(id="b", columns=("c1",), text="1")),
            ),
            Row(
                id="r1",
                cells=(
                    Cell(id="c", columns=("c0",), text="Duties", level=0),
                    Cell(id="d", columns=("c1",), text="2", markers=(Marker(at=1, text="1", note_ref="n0"),)),
                ),
            ),
        ),
        notes=(Note(id="n0", label="1/", text="Provisional."),),
    )
    return replace(table, **changes)  # type: ignore[arg-type]


def with_rows(*rows: Row, **changes: object) -> Table:
    return base(rows=rows, **changes)


def codes(issues: tuple[Issue, ...]) -> list[str]:
    return [i.code for i in issues]


def single(issues: tuple[Issue, ...], code: CoreIssue, object_id: str | None) -> Issue:
    assert codes(issues) == [code]
    assert issues[0].object_id == object_id
    return issues[0]


# Valid tables


def test_base_table_is_valid() -> None:
    assert validate_table(base()) == ()


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.name)
def test_examples_are_valid(path: Path) -> None:
    assert validate_table(decode_table(json.loads(path.read_text(encoding="utf-8")))) == ()


def test_proposed_spec_example_is_valid() -> None:
    html = """# Receipts (in dollars)
<table>
  <thead><tr><th>Item</th><th>2025</th></tr></thead>
  <tbody>
    <tr><td data-level="0">Receipts:</td><td></td></tr>
    <tr><td data-level="1">Taxes</td><td>100<sup>1</sup></td></tr>
  </tbody>
  <tfoot><tr><td colspan="2">1/ Provisional figure.</td></tr></tfoot>
</table>"""
    parsed = parse_output(html).fragments[0].table
    assert parsed is not None
    table = propose_table(parsed, table_id="t").table
    assert table is not None
    assert validate_table(table) == ()


def test_merged_cells_and_header_only_tables_are_valid() -> None:
    merged = with_rows(
        Row(
            id="r0",
            cells=(
                Cell(id="a", columns=("c0",), text="x", rowspan=2, level=0),
                Cell(id="b", columns=("c1",), text="1"),
            ),
        ),
        Row(id="r1", cells=(Cell(id="d", columns=("c1",), text="2"),)),
    )
    assert validate_table(merged) == ()
    assert validate_table(with_rows()) == ()


def test_every_issue_blocks_and_comes_from_validation() -> None:
    table = base(
        headers=(Header(id="h0", text="Item"), Header(id="h1", text="2025"), Header(id="h9", text="x")), label_columns=2
    )
    issues = validate_table(table)
    assert codes(issues) == [CoreIssue.UNREFERENCED_HEADER, CoreIssue.LABEL_COLUMNS_OUT_OF_RANGE]
    assert all(i.blocking and i.stage is Stage.VALIDATION and i.source is None for i in issues)


# 1. IDs and references


def test_duplicate_id_across_kinds() -> None:
    table = base(notes=(Note(id="h0", label="1/", text="Provisional."),))
    issues = validate_table(table)
    # the note's ID repeats a header ID; the marker linking n0 now dangles
    assert codes(issues) == [CoreIssue.DUPLICATE_ID, CoreIssue.DANGLING_REFERENCE]
    assert issues[0].object_id == "h0"
    assert "header" in issues[0].message


def test_dangling_header_reference() -> None:
    table = base(columns=(Column(id="c0", headers=("h0",)), Column(id="c1", headers=("h1", "h7"))))
    single(validate_table(table), CoreIssue.DANGLING_REFERENCE, "c1")


def test_dangling_column_reference_does_not_cascade() -> None:
    table = with_rows(
        Row(id="r0", cells=(Cell(id="a", columns=("c0",), text="x", level=0), Cell(id="b", columns=("c9",), text="1"))),
    )
    issue = single(validate_table(table), CoreIssue.DANGLING_REFERENCE, "b")
    assert "column 'c9'" in issue.message


@pytest.mark.parametrize(
    ("changes", "object_id"),
    [
        ({"note_refs": ("n5",)}, None),
        ({"title": TextBlock(text="R", markers=(Marker(at=1, text="2", note_ref="n5"),))}, None),
        ({"title": TextBlock(text="R", note_refs=("n5",))}, None),
        ({"headers": (Header(id="h0", text="Item", note_refs=("n5",)), Header(id="h1", text="2025"))}, "h0"),
    ],
)
def test_dangling_note_references(changes: dict[str, object], object_id: str | None) -> None:
    single(validate_table(base(**changes)), CoreIssue.DANGLING_REFERENCE, object_id)


def test_dangling_marker_link_in_a_cell() -> None:
    table = base(notes=())
    single(validate_table(table), CoreIssue.DANGLING_REFERENCE, "d")


# 2. Header paths


def headed(columns: tuple[Column, ...], headers: tuple[str, ...]) -> Table:
    """A one-row table with the given columns and headers."""
    cells = tuple(
        Cell(id=f"x{i}", columns=(c.id,), text="1", level=0 if i == 0 else None) for i, c in enumerate(columns)
    )
    return Table(
        id="t",
        headers=tuple(Header(id=h, text=h) for h in headers),
        columns=columns,
        rows=(Row(id="r0", cells=cells),),
    )


def test_shared_header_under_different_parents_is_fine() -> None:
    # a units line under the headings, spanning columns that have different headers above them
    table = headed(
        (Column(id="c0", headers=("a", "units")), Column(id="c1", headers=("b", "units"))),
        ("a", "b", "units"),
    )
    assert validate_table(table) == ()


def test_shared_header_on_columns_that_are_not_adjacent() -> None:
    table = headed(
        (Column(id="c0", headers=("a",)), Column(id="c1", headers=("b",)), Column(id="c2", headers=("a",))),
        ("a", "b"),
    )
    single(validate_table(table), CoreIssue.HEADER_STRUCTURE, "a")


def test_unreferenced_header() -> None:
    table = headed((Column(id="c0", headers=()),), ("a",))
    single(validate_table(table), CoreIssue.UNREFERENCED_HEADER, "a")


def test_paths_of_different_lengths_are_fine() -> None:
    table = headed((Column(id="c0", headers=("a",)), Column(id="c1", headers=("b", "c"))), ("a", "b", "c"))
    assert validate_table(table) == ()


# 3. Cells and grid coverage


def test_uncovered_positions_are_reported_once_per_row() -> None:
    table = base(
        columns=(Column(id="c0", headers=("h0",)), Column(id="c1", headers=("h1",)), Column(id="c2", headers=("h1",)))
    )
    issues = validate_table(table)
    assert codes(issues) == [CoreIssue.UNCOVERED_POSITION, CoreIssue.UNCOVERED_POSITION]
    assert [(i.object_id, i.message) for i in issues] == [
        ("r0", "row 'r0': no cell covers c2"),
        ("r1", "row 'r1': no cell covers c2"),
    ]


def test_overlapping_cells_are_reported_on_the_later_cell() -> None:
    table = with_rows(
        Row(
            id="r0",
            cells=(
                Cell(id="a", columns=("c0",), text="x", level=0),
                Cell(id="b", columns=("c1",), text="1", rowspan=2),
            ),
        ),
        Row(id="r1", cells=(Cell(id="c", columns=("c0", "c1"), text="y", level=0),)),
    )
    single(validate_table(table), CoreIssue.OVERLAPPING_CELLS, "c")


def test_rowspan_out_of_bounds() -> None:
    table = with_rows(
        Row(
            id="r0",
            cells=(
                Cell(id="a", columns=("c0",), text="x", level=0, rowspan=2),
                Cell(id="b", columns=("c1",), text="1"),
            ),
        ),
    )
    single(validate_table(table), CoreIssue.ROWSPAN_OUT_OF_BOUNDS, "a")


def test_cell_order() -> None:
    table = with_rows(
        Row(id="r0", cells=(Cell(id="b", columns=("c1",), text="1"), Cell(id="a", columns=("c0",), text="x", level=0))),
    )
    single(validate_table(table), CoreIssue.CELL_ORDER, "a")


def test_nonconsecutive_columns() -> None:
    table = base(
        columns=(Column(id="c0", headers=("h0",)), Column(id="c1", headers=("h1",)), Column(id="c2", headers=("h1",))),
        rows=(
            Row(
                id="r0",
                cells=(Cell(id="a", columns=("c0",), text="x", level=0), Cell(id="b", columns=("c2", "c1"), text="1")),
            ),
        ),
    )
    single(validate_table(table), CoreIssue.NONCONSECUTIVE_COLUMNS, "b")


# 4. Levels and label columns


def test_level_outside_the_first_column() -> None:
    table = with_rows(
        Row(
            id="r0",
            cells=(Cell(id="a", columns=("c0",), text="x", level=0), Cell(id="b", columns=("c1",), text="1", level=1)),
        ),
    )
    single(validate_table(table), CoreIssue.MISPLACED_LEVEL, "b")


def test_level_on_a_cell_spanning_from_the_first_column() -> None:
    table = with_rows(Row(id="r0", cells=(Cell(id="a", columns=("c0", "c1"), text="Receipts", level=0),)))
    assert validate_table(table) == ()


@pytest.mark.parametrize(("label_columns", "valid"), [(1, True), (2, False), (3, False)])
def test_label_columns_must_leave_a_value_column(label_columns: int, valid: bool) -> None:
    issues = validate_table(base(label_columns=label_columns))
    assert codes(issues) == ([] if valid else [CoreIssue.LABEL_COLUMNS_OUT_OF_RANGE])


def test_single_column_table_may_be_all_labels() -> None:
    table = Table(
        id="t",
        headers=(),
        columns=(Column(id="c0", headers=()),),
        rows=(Row(id="r0", cells=(Cell(id="a", columns=("c0",), text="x", level=0),)),),
    )
    assert validate_table(table) == ()


# 5. Marker positions


@pytest.mark.parametrize(
    "markers",
    [
        (Marker(at=5, text="1", note_ref="n0"),),  # text "2" has length 1
        (Marker(at=1, text="1", note_ref="n0"), Marker(at=0, text="2")),
    ],
)
def test_invalid_marker_positions(markers: tuple[Marker, ...]) -> None:
    table = with_rows(
        Row(
            id="r0",
            cells=(
                Cell(id="a", columns=("c0",), text="x", level=0),
                Cell(id="d", columns=("c1",), text="2", markers=markers),
            ),
        ),
    )
    single(validate_table(table), CoreIssue.INVALID_MARKER_POSITION, "d")


def test_markers_at_the_same_position_are_fine() -> None:
    markers = (Marker(at=1, text="1", note_ref="n0"), Marker(at=1, text="2"))
    table = with_rows(
        Row(
            id="r0",
            cells=(
                Cell(id="a", columns=("c0",), text="x", level=0),
                Cell(id="d", columns=("c1",), text="2", markers=markers),
            ),
        ),
    )
    assert validate_table(table) == ()


def test_invalid_marker_position_in_the_title() -> None:
    table = base(title=TextBlock(text="R", markers=(Marker(at=2, text="1", note_ref="n0"),)))
    single(validate_table(table), CoreIssue.INVALID_MARKER_POSITION, None)


# 6. note_refs


def test_note_ref_duplicating_a_marker_link() -> None:
    cell = Cell(id="d", columns=("c1",), text="2", markers=(Marker(at=1, text="1", note_ref="n0"),), note_refs=("n0",))
    table = with_rows(Row(id="r0", cells=(Cell(id="a", columns=("c0",), text="x", level=0), cell)))
    single(validate_table(table), CoreIssue.DUPLICATE_NOTE_REF, "d")


def test_note_refs_and_unreferenced_notes_are_fine() -> None:
    table = base(
        note_refs=("n1",),
        notes=(
            Note(id="n0", label="1/", text="Provisional."),
            Note(id="n1", text="General."),
            Note(id="n2", text="Unused."),
        ),
    )
    assert validate_table(table) == ()


# 7. Unicode


def test_unpaired_surrogate() -> None:
    table = with_rows(
        Row(
            id="r0",
            cells=(Cell(id="a", columns=("c0",), text="x\ud800", level=0), Cell(id="b", columns=("c1",), text="1")),
        ),
    )
    single(validate_table(table), CoreIssue.INVALID_UNICODE, "a")


def test_astral_characters_are_fine() -> None:
    table = base(title=TextBlock(text="A\N{GRINNING FACE}", markers=(Marker(at=2, text="1", note_ref="n0"),)))
    assert validate_table(table) == ()


# Order


def test_issue_order_follows_the_checklist() -> None:
    table = base(
        title=TextBlock(text="R\udc00"),
        label_columns=2,
        columns=(
            Column(id="c0", headers=("h0",)),
            Column(id="c1", headers=("h1",)),
            Column(id="c2", headers=("h1", "h0")),
        ),
    )
    # h0 is shared by c0 and c2, which are not adjacent; nothing covers c2
    assert codes(validate_table(table)) == [
        CoreIssue.HEADER_STRUCTURE,
        CoreIssue.UNCOVERED_POSITION,
        CoreIssue.UNCOVERED_POSITION,
        CoreIssue.INVALID_UNICODE,
    ]
    assert validate_table(table) == validate_table(table)
