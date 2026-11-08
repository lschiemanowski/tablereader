"""The record metric and grid comparison, on small transcriptions."""

import pytest

from evaluation.grids import Grid, GridCell, compare_grids, grid_from_parsed, normalize_cell
from evaluation.records import NOT_A_TABLE, crop_score, normalize_label, table_records
from table_core.models import Table
from table_core.parsing import ParsedTable, parse_output
from table_core.structure import propose_table


def parsed(text: str) -> list[ParsedTable]:
    tables = [f.table for f in parse_output(text).fragments]
    assert all(t is not None for t in tables)
    return [t for t in tables if t is not None]


def tables(text: str) -> list[Table]:
    out: list[Table] = []
    for i, p in enumerate(parsed(text)):
        table = propose_table(p, table_id=f"t{i}").table
        assert table is not None
        out.append(table)
    return out


HIERARCHY = """<table>
<thead><tr><th></th><th colspan="2">2025</th></tr><tr><th></th><th>Actual</th><th>Budget</th></tr></thead>
<tbody>
<tr><td data-level="0">Receipts:</td><td></td><td></td></tr>
<tr><td data-level="1">Taxes</td><td>100<sup>1</sup></td><td>90</td></tr>
<tr><td data-level="1">Duties</td><td colspan="2">Not disclosed</td></tr>
</tbody>
<tfoot><tr><td colspan="3">1/ Provisional.</td></tr></tfoot>
</table>"""


# Records


def test_records_carry_row_path_column_path_and_value() -> None:
    (table,) = tables(HIERARCHY)
    result = table_records(table)
    assert result.records == {
        (("receipts", "taxes"), ("2025", "actual"), "100"),
        (("receipts", "taxes"), ("2025", "budget"), "90"),
        (("receipts", "duties"), ("2025", "actual"), "Not disclosed"),  # a merged value is one record
    }
    assert (result.level_fallback_rows, result.duplicates) == (0, 0)


def test_missing_level_falls_back_to_zero_and_is_counted() -> None:
    (table,) = tables(
        "<table><tbody>"
        '<tr><td data-level="0">A:</td><td></td></tr>'
        '<tr><td data-level="1">B:</td><td></td></tr>'
        "<tr><td>x</td><td>1</td></tr>"  # under B: or only under A:? unresolved; level 0 closes B:
        "</tbody></table>"
    )
    result = table_records(table)
    assert result.level_fallback_rows == 1
    assert result.records == {(("a", "x"), (), "1")}


def test_duplicate_records_are_counted() -> None:
    row = '<tr><td data-level="0">x</td><td>1</td></tr>'
    (table,) = tables(f"<table><tbody>{row}{row}</tbody></table>")
    assert table_records(table).duplicates == 1


@pytest.mark.parametrize(
    ("label", "normalized"),
    [
        ("Receipts:", "receipts"),
        ("Total....... ", "total"),
        ("1945 \N{EN DASH} 1946", "1945-1946"),
        ("Rate  of  Interest", "rate of interest"),
        ("2 ½ percent", "2 1/2 percent"),
    ],
)
def test_normalize_label(label: str, normalized: str) -> None:
    assert normalize_label(label) == normalized


def test_crop_score() -> None:
    reference = tables(HIERARCHY)
    assert crop_score(reference, tables(HIERARCHY)) == 1.0
    changed = tables(HIERARCHY.replace(">90<", ">91<"))
    assert crop_score(reference, changed) == pytest.approx(2 / 4)
    assert crop_score(reference, None) == 0.0
    assert crop_score(reference, NOT_A_TABLE) == 0.0
    assert crop_score(NOT_A_TABLE, NOT_A_TABLE) == 1.0
    assert crop_score(NOT_A_TABLE, reference) == 0.0
    with pytest.raises(ValueError):
        crop_score(None, reference)


def test_records_ignore_markers_and_titles() -> None:
    reference = tables(HIERARCHY)
    other = tables("# Some title\n" + HIERARCHY.replace("100<sup>1</sup>", "100"))
    assert crop_score(reference, other) == 1.0


# Grids


def test_grid_from_transcription_stacks_head_and_body_and_puts_markers_back() -> None:
    grid = grid_from_parsed(parsed(HIERARCHY))
    assert (grid.rows, grid.columns) == (5, 3)
    assert GridCell(0, 1, 1, 2, "2025") in grid.cells
    assert GridCell(3, 1, 1, 1, "1001") in grid.cells
    assert GridCell(4, 1, 1, 2, "Not disclosed") in grid.cells
    assert all("Provisional" not in c.text for c in grid.cells)


def test_two_tables_stack_in_reading_order() -> None:
    one = "<table><tbody><tr><td>a</td><td>1</td></tr></tbody></table>"
    two = "<table><tbody><tr><td>b</td><td>2</td><td>3</td></tr></tbody></table>"
    grid = grid_from_parsed(parsed(one + "\n" + two))
    assert (grid.rows, grid.columns) == (2, 3)
    assert GridCell(1, 0, 1, 1, "b") in grid.cells


@pytest.mark.parametrize(("a", "b"), [("Providers 32", "Providers³²"), ("Antisemi\xad tismus", "Antisemitismus")])
def test_normalize_cell(a: str, b: str) -> None:
    assert normalize_cell(a) == normalize_cell(b)


def test_compare_grids() -> None:
    reference = Grid(
        rows=2,
        columns=2,
        cells=(GridCell(0, 0, 1, 1, "Item"), GridCell(0, 1, 1, 1, "2025"), GridCell(1, 0, 1, 1, "Taxes"),
               GridCell(1, 1, 1, 1, "100 1")),
    )  # fmt: skip
    same = grid_from_parsed(
        parsed(
            "<table><thead><tr><th>Item</th><th>2025</th></tr></thead><tbody><tr><td>Taxes</td><td>100<sup>1</sup></td></tr></tbody></table>"
        )
    )
    result = compare_grids(reference, same)
    assert (result.same_shape, result.structure_f1, result.text_f1, result.content_f1) == (True, 1.0, 1.0, 1.0)
    shifted = grid_from_parsed(
        parsed(
            "<table><tbody><tr><td>Note</td><td></td></tr><tr><td>Item</td><td>2025</td></tr><tr><td>Taxes</td><td>100<sup>1</sup></td></tr></tbody></table>"
        )
    )
    result = compare_grids(reference, shifted)
    assert not result.same_shape
    assert result.text_f1 < 0.5
    assert result.content_f1 == pytest.approx(2 * 4 / 9)
