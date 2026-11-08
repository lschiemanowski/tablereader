"""Deriving column and row paths."""

import json
from pathlib import Path

from table_core.codec import decode_table
from table_core.models import Table
from table_core.parsing import parse_output
from table_core.paths import RowPath, column_paths, row_paths
from table_core.structure import propose_table

EXAMPLES = Path(__file__).parents[3] / "examples"


def example(name: str) -> Table:
    return decode_table(json.loads((EXAMPLES / name).read_text(encoding="utf-8")))


def table(body: str, head: str = "") -> Table:
    html = "<table>" + (f"<thead>{head}</thead>" if head else "") + f"<tbody>{body}</tbody></table>"
    output = parse_output(html)
    parsed = output.fragments[0].table
    assert parsed is not None
    proposal = propose_table(parsed, table_id="t")
    assert proposal.table is not None
    return proposal.table


def rows(*specs: tuple[int | None, str, str]) -> str:
    """Body rows from (level, label, value); level None writes no data-level."""
    out: list[str] = []
    for level, label, value in specs:
        attr = "" if level is None else f' data-level="{level}"'
        out.append(f"<tr><td{attr}>{label}</td><td>{value}</td></tr>")
    return "".join(out)


def labels(t: Table) -> list[tuple[str, ...] | None]:
    return [p.labels for p in row_paths(t)]


# Column paths


def test_column_paths_of_the_examples() -> None:
    assert column_paths(example("multilevel-notes.json")) == {
        "c0": ("Product",),
        "c1": ("Revenue", "2024", "Actual"),
        "c2": ("Revenue", "2024", "Budget"),
        "c3": ("Revenue", "2025", "Actual"),
        "c4": ("Revenue", "2025", "Budget"),
    }
    assert column_paths(example("duplicate-headers.json")) == {"c0": ("Amount",), "c1": ("Amount",)}


def test_blank_headers_are_skipped() -> None:
    t = table(
        rows((0, "A", "1")).replace("<td>1</td>", "<td>1</td><td>2</td>"),
        head='<tr><th rowspan="2"></th><th colspan="2">2025</th></tr><tr><th>Actual</th><th>Budget</th></tr>',
    )
    assert column_paths(t) == {"c0": (), "c1": ("2025", "Actual"), "c2": ("2025", "Budget")}


def test_equal_texts_from_distinct_headers_are_kept() -> None:
    t = table(rows((0, "A", "1")), head="<tr><th>Date</th><th>Bonds</th></tr><tr><th>Date</th><th>Bonds</th></tr>")
    assert column_paths(t) == {"c0": ("Date", "Date"), "c1": ("Bonds", "Bonds")}


def test_columns_without_headers() -> None:
    assert column_paths(table(rows((0, "A", "1")))) == {"c0": (), "c1": ()}


# Row paths: the heading rule


def test_row_hierarchy_example() -> None:
    assert row_paths(example("row-hierarchy.json")) == (
        RowPath(row_id="r0", labels=("Receipts:",), heading=True),
        RowPath(row_id="r1", labels=("Receipts:", "Taxes"), heading=False),
        RowPath(row_id="r2", labels=("Receipts:", "Duties"), heading=False),
        RowPath(row_id="r3", labels=("Expenses:",), heading=True),
        RowPath(row_id="r4", labels=("Expenses:", "Wages"), heading=False),
    )


def test_two_label_columns_with_a_merged_label() -> None:
    assert labels(example("merged-cells.json")) == [("Germany", "Manufacturing"), ("Germany", "Services")]


def test_nested_headings_and_closing() -> None:
    t = table(
        rows(
            (0, "Receipts:", ""),
            (1, "Taxes:", ""),
            (2, "Income", "100"),
            (1, "Customs", "20"),  # a value row keeps headings at its own level open
            (1, "Other:", ""),  # a new heading closes the heading at its level
            (2, "Fees", "5"),
            (0, "Total", "125"),
            (1, "Memo", "1"),
        )
    )
    assert labels(t) == [
        ("Receipts:",),
        ("Receipts:", "Taxes:"),
        ("Receipts:", "Taxes:", "Income"),
        ("Receipts:", "Taxes:", "Customs"),
        ("Receipts:", "Other:"),
        ("Receipts:", "Other:", "Fees"),
        ("Receipts:", "Total"),  # a value row at level 0 keeps a level-0 heading open
        ("Receipts:", "Memo"),
    ]
    assert [p.heading for p in row_paths(t)] == [True, True, False, False, True, False, False, False]


def test_dash_is_value_text() -> None:
    t = table(rows((0, "A", "-"), (1, "B", "1")))
    assert labels(t) == [("A",), ("B",)]


def test_label_spanning_into_value_columns_is_a_heading() -> None:
    t = table('<tr><td data-level="0" colspan="2">Receipts</td></tr>' + rows((1, "Taxes:", ""), (2, "Income", "100")))
    assert labels(t) == [("Receipts",), ("Receipts", "Taxes:"), ("Receipts", "Taxes:", "Income")]
    assert row_paths(t)[0].heading


def test_spanning_label_with_other_value_text_is_a_value_row() -> None:
    t = table(
        '<tr><td data-level="0" colspan="2">Receipts</td><td>9</td></tr>'
        '<tr><td data-level="1">Taxes</td><td>1</td><td>2</td></tr>'
    )
    assert [p.heading for p in row_paths(t)] == [False, False]
    assert labels(t) == [("Receipts",), ("Taxes",)]


def test_value_spanning_from_above_counts_as_value_text() -> None:
    t = table(
        '<tr><td data-level="0">Series E</td><td rowspan="2">2.9%</td></tr>'
        '<tr><td data-level="0">Series F</td></tr>'
        '<tr><td data-level="1">Group 1</td><td>40</td></tr>'
    )
    assert labels(t) == [("Series E",), ("Series F",), ("Group 1",)]
    assert not any(p.heading for p in row_paths(t))


def test_blank_row_leaves_open_headings_alone() -> None:
    t = table(rows((0, "Receipts:", ""), (1, "Taxes:", ""), (2, "Income", "100"), (0, "", ""), (2, "Corporate", "50")))
    assert labels(t) == [
        ("Receipts:",),
        ("Receipts:", "Taxes:"),
        ("Receipts:", "Taxes:", "Income"),
        ("Receipts:", "Taxes:"),
        ("Receipts:", "Taxes:", "Corporate"),
    ]
    assert not row_paths(t)[3].heading


def test_blank_row_without_level_does_not_unresolve() -> None:
    t = table(rows((0, "A:", ""), (None, "", ""), (1, "B", "1")))
    assert labels(t) == [("A:",), ("A:",), ("A:", "B")]


def test_row_with_empty_label_and_values() -> None:
    t = table(rows((0, "A:", ""), (1, "", "5")))
    assert labels(t) == [("A:",), ("A:",)]


# Row paths: missing levels


def test_missing_level_on_a_value_row() -> None:
    t = table(
        rows(
            (0, "Receipts:", ""),
            (1, "Taxes:", ""),
            (2, "Income", "100"),
            (None, "Estate", "5"),  # level 2 or 0? its own path is unknown
            (2, "Corporate", "50"),  # Estate may have closed Taxes:
            (0, "Expenditures:", ""),  # a level-0 heading closes everything: resolved again
            (1, "Defense", "70"),
        )
    )
    assert labels(t) == [
        ("Receipts:",),
        ("Receipts:", "Taxes:"),
        ("Receipts:", "Taxes:", "Income"),
        None,
        None,
        ("Expenditures:",),
        ("Expenditures:", "Defense"),
    ]


def test_value_row_of_unknown_level_keeps_level_0_headings() -> None:
    t = table(rows((0, "Receipts:", ""), (None, "Taxes", "1"), (1, "Duties", "2")))
    assert labels(t) == [("Receipts:",), ("Receipts:", "Taxes"), ("Receipts:", "Duties")]


def test_heading_of_unknown_level() -> None:
    t = table(
        rows(
            (0, "Receipts:", ""),
            (None, "Taxes:", ""),  # closes Receipts: if at level 0
            (2, "Income", "1"),  # under Taxes:, but is Receipts: still open?
            (0, "Expenditures:", ""),
            (1, "Defense", "2"),
        )
    )
    assert labels(t) == [("Receipts:",), None, None, ("Expenditures:",), ("Expenditures:", "Defense")]


def test_heading_of_unknown_level_with_nothing_open_is_resolved() -> None:
    t = table(rows((None, "Receipts:", ""), (0, "Total", "1")))
    # Total at level 0 keeps Receipts: open only if Receipts: is at level 0 too.
    assert labels(t) == [("Receipts:",), None]


def test_missing_level_without_open_headings_is_resolved() -> None:
    t = table(rows((None, "A", "1"), (None, "B", "2")))
    assert labels(t) == [("A",), ("B",)]


def test_uncertainty_ends_when_the_heading_is_closed_for_sure() -> None:
    t = table(
        rows(
            (0, "A:", ""),
            (1, "B:", ""),
            (None, "x", "1"),  # may close B:
            (1, "C:", ""),  # closes B: (level 1 is not < 1) whatever x did
            (2, "y", "2"),
        )
    )
    assert labels(t) == [("A:",), ("A:", "B:"), None, ("A:", "C:"), ("A:", "C:", "y")]


def test_paths_do_not_include_marker_text() -> None:
    t = table(rows((0, "Receipts:<sup>1</sup>", ""), (1, "Taxes", "1")))
    assert labels(t) == [("Receipts:",), ("Receipts:", "Taxes")]
