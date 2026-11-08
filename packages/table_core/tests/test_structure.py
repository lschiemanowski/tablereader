"""Proposing table JSON from parsed transcriptions."""

import pytest

from table_core.codec import decode_table, encode_table, schema_errors
from table_core.issues import CoreIssue, Issue, Section, SourceRef
from table_core.models import Cell, Column, Header, Marker, Note, Row, TextBlock
from table_core.parsing import ParsedRow, ParsedTable, parse_output
from table_core.structure import TableProposal, normalize_note_label, propose_table, split_note_label

EXAMPLE = """# Receipts (in dollars)
<table>
  <thead>
    <tr><th>Item</th><th>2025</th></tr>
  </thead>
  <tbody>
    <tr><td data-level="0">Receipts:</td><td></td></tr>
    <tr><td data-level="1">Taxes</td><td>100<sup>1</sup></td></tr>
  </tbody>
  <tfoot>
    <tr><td colspan="2">1/ Provisional figure.</td></tr>
  </tfoot>
</table>"""


def parsed(text: str) -> ParsedTable:
    output = parse_output(text)
    assert len(output.fragments) == 1
    table = output.fragments[0].table
    assert table is not None
    return table


def propose(text: str) -> TableProposal:
    return propose_table(parsed(text), table_id="t")


def table_html(*, head: str = "", body: str = "", foot: str = "", title: str = "") -> str:
    sections = "".join(
        f"<{tag}>{rows}</{tag}>" for tag, rows in (("thead", head), ("tbody", body), ("tfoot", foot)) if rows
    )
    return (f"# {title}\n" if title else "") + f"<table>{sections}</table>"


def codes(issues: tuple[Issue, ...]) -> list[str]:
    return [i.code for i in issues]


def paths(proposal: TableProposal) -> list[list[str]]:
    """Each column's header path as texts."""
    assert proposal.table is not None
    text = {h.id: h.text for h in proposal.table.headers}
    return [[text[h] for h in column.headers] for column in proposal.table.columns]


def cells(proposal: TableProposal) -> list[list[tuple[str, tuple[str, ...], int]]]:
    """Each row's cells as (text, columns, rowspan)."""
    assert proposal.table is not None
    return [[(c.text, c.columns, c.rowspan) for c in row.cells] for row in proposal.table.rows]


# The spec example


def test_example_from_the_spec() -> None:
    proposal = propose(EXAMPLE)
    assert proposal.issues == ()
    table = proposal.table
    assert table is not None
    assert table.id == "t"
    assert table.title == TextBlock(text="Receipts (in dollars)")
    assert table.headers == (Header(id="h0", text="Item"), Header(id="h1", text="2025"))
    assert table.columns == (Column(id="c0", headers=("h0",)), Column(id="c1", headers=("h1",)))
    assert table.rows == (
        Row(
            id="r0",
            cells=(
                Cell(id="r0c0", columns=("c0",), text="Receipts:", level=0),
                Cell(id="r0c1", columns=("c1",), text=""),
            ),
        ),
        Row(
            id="r1",
            cells=(
                Cell(id="r1c0", columns=("c0",), text="Taxes", level=1),
                Cell(id="r1c1", columns=("c1",), text="100", markers=(Marker(at=3, text="1", note_ref="n0"),)),
            ),
        ),
    )
    assert table.notes == (Note(id="n0", label="1/", text="Provisional figure."),)
    assert table.label_columns == 1


def test_sources_point_at_the_transcription() -> None:
    sources = propose(EXAMPLE).sources
    assert sources["h1"] == SourceRef(fragment=0, section=Section.HEAD, row=0, cell=1)
    assert sources["r1"] == SourceRef(fragment=0, section=Section.BODY, row=1)
    assert sources["r1c1"] == SourceRef(fragment=0, section=Section.BODY, row=1, cell=1)
    assert sources["n0"] == SourceRef(fragment=0, section=Section.FOOT, row=0)
    assert "c0" not in sources


def test_proposal_encodes_to_schema_valid_json() -> None:
    table = propose(EXAMPLE).table
    assert table is not None
    payload = encode_table(table)
    assert schema_errors(payload) == ()
    assert decode_table(payload) == table


def test_same_input_gives_same_proposal() -> None:
    assert propose(EXAMPLE) == propose(EXAMPLE)


# Header paths


MULTILEVEL = table_html(
    head='<tr><th rowspan="2">Item</th><th colspan="2">2025</th><th rowspan="2">Total</th></tr>'
    "<tr><th>Actual</th><th>Budget</th></tr>",
    body='<tr><td data-level="0">A</td><td>1</td><td>2</td><td>3</td></tr>',
)


def test_header_spans_become_shared_headers_and_paths() -> None:
    proposal = propose(MULTILEVEL)
    assert proposal.issues == ()
    assert paths(proposal) == [["Item"], ["2025", "Actual"], ["2025", "Budget"], ["Total"]]
    assert proposal.table is not None
    assert [c.headers for c in proposal.table.columns] == [("h0",), ("h1", "h3"), ("h1", "h4"), ("h2",)]


def test_equal_header_texts_keep_distinct_ids() -> None:
    proposal = propose(
        table_html(
            head="<tr><th>A</th><th>A</th></tr><tr><th>A</th><th>B</th></tr>",
            body='<tr><td data-level="0">x</td><td>1</td></tr>',
        )
    )
    assert proposal.table is not None
    assert [c.headers for c in proposal.table.columns] == [("h0", "h2"), ("h1", "h3")]


def test_blank_header_cell_is_a_header_with_empty_text() -> None:
    proposal = propose(
        table_html(head="<tr><th></th><th>2025</th></tr>", body='<tr><td data-level="0">A</td><td>1</td></tr>')
    )
    assert proposal.table is not None
    assert proposal.table.headers[0] == Header(id="h0", text="")
    assert paths(proposal) == [[""], ["2025"]]


def test_table_without_thead_has_columns_without_headers() -> None:
    proposal = propose(table_html(body='<tr><td data-level="0">Wheat</td><td>1,204</td><td>1,310</td></tr>'))
    assert proposal.issues == ()
    assert proposal.table is not None
    assert proposal.table.headers == ()
    assert [c.headers for c in proposal.table.columns] == [(), (), ()]


def test_header_only_table_has_no_rows() -> None:
    proposal = propose(table_html(head="<tr><th>A</th><th>B</th></tr>"))
    assert proposal.issues == ()
    assert proposal.table is not None
    assert proposal.table.rows == ()
    assert paths(proposal) == [["A"], ["B"]]


def test_header_marker_and_unclear_are_kept() -> None:
    proposal = propose(
        table_html(
            head='<tr><th>Item</th><th data-unclear="1">Rate<sup>a</sup></th></tr>',
            body='<tr><td data-level="0">x</td><td>1</td></tr>',
            foot='<tr><td colspan="2">a/ Annual.</td></tr>',
        )
    )
    assert proposal.issues == ()
    assert proposal.table is not None
    assert proposal.table.headers[1] == Header(
        id="h1", text="Rate", markers=(Marker(at=4, text="a", note_ref="n0"),), unclear=True
    )


# Grids that do not add up


def test_short_body_rows_are_not_padded() -> None:
    proposal = propose(
        table_html(
            head="<tr><th>A</th><th>B</th><th>C</th></tr>",
            body='<tr><td data-level="0">x</td><td>1</td></tr>',
        )
    )
    assert proposal.issues == ()  # the uncovered body position is left to validation
    assert proposal.table is not None
    assert len(proposal.table.columns) == 3
    assert cells(proposal) == [[("x", ("c0",), 1), ("1", ("c1",), 1)]]


def test_header_narrower_than_body_is_a_blocking_gap() -> None:
    proposal = propose(
        table_html(head="<tr><th>A</th><th>B</th></tr>", body='<tr><td data-level="0">x</td><td>1</td><td>2</td></tr>')
    )
    assert codes(proposal.issues) == [CoreIssue.UNCOVERED_POSITION]
    issue = proposal.issues[0]
    assert issue.blocking
    assert issue.source == SourceRef(fragment=0, section=Section.HEAD, row=0)
    assert "column(s) 2 of 3" in issue.message
    assert paths(proposal) == [["A"], ["B"], []]


def test_header_spans_that_overflow_leave_gaps_in_other_rows() -> None:
    proposal = propose(
        table_html(
            head='<tr><th rowspan="2">Item</th><th colspan="3">Receipts</th></tr><tr><th>a</th><th>b</th></tr>',
            body='<tr><td data-level="0">x</td><td>1</td><td>2</td><td>3</td></tr>',
        )
    )
    assert codes(proposal.issues) == [CoreIssue.UNCOVERED_POSITION]
    assert proposal.issues[0].source == SourceRef(fragment=0, section=Section.HEAD, row=1)


def test_empty_header_row_not_covered_by_rowspans_is_a_gap() -> None:
    proposal = propose(
        table_html(head="<tr><th>A</th><th>B</th></tr><tr></tr>", body='<tr><td data-level="0">x</td><td>1</td></tr>')
    )
    assert codes(proposal.issues) == [CoreIssue.UNCOVERED_POSITION]


def test_empty_header_row_covered_by_rowspans_is_fine() -> None:
    proposal = propose(
        table_html(
            head='<tr><th rowspan="2">A</th><th rowspan="2">B</th></tr><tr></tr>',
            body='<tr><td data-level="0">x</td><td>1</td></tr>',
        )
    )
    assert proposal.issues == ()
    assert paths(proposal) == [["A"], ["B"]]


def test_overlapping_header_cells_are_blocking_and_the_earlier_cell_wins() -> None:
    # B spans down into row 1, column 1; C in row 1 starts at column 0 and spans two columns across it.
    proposal = propose(
        table_html(
            head='<tr><th>A</th><th rowspan="2">B</th></tr><tr><th colspan="2">C</th></tr>',
            body='<tr><td data-level="0">x</td><td>1</td></tr>',
        )
    )
    assert codes(proposal.issues) == [CoreIssue.OVERLAPPING_CELLS]
    issue = proposal.issues[0]
    assert (issue.blocking, issue.object_id) == (True, "h2")
    assert paths(proposal) == [["A", "C"], ["B"]]


@pytest.mark.parametrize("section", ["head", "body"])
def test_rowspan_past_the_section_end_is_shortened(section: str) -> None:
    if section == "head":
        html = table_html(
            head='<tr><th rowspan="3">A</th><th>B</th></tr>', body='<tr><td data-level="0">x</td><td>1</td></tr>'
        )
    else:
        html = table_html(body='<tr><td data-level="0" rowspan="3">x</td><td>1</td></tr><tr><td>2</td></tr>')
    proposal = propose(html)
    assert codes(proposal.issues) == [CoreIssue.UNEXPECTED_MARKUP]
    assert not proposal.issues[0].blocking
    if section == "body":
        assert cells(proposal)[0][0] == ("x", ("c0",), 2)


# Body cells


def test_body_spans() -> None:
    proposal = propose(
        table_html(
            body='<tr><td data-level="0" rowspan="2">x</td><td colspan="2">wide</td></tr><tr><td>1</td><td>2</td></tr>',
        )
    )
    assert proposal.issues == ()
    assert cells(proposal) == [
        [("x", ("c0",), 2), ("wide", ("c1", "c2"), 1)],
        [("1", ("c1",), 1), ("2", ("c2",), 1)],
    ]
    assert proposal.table is not None
    assert [c.id for c in proposal.table.rows[1].cells] == ["r1c1", "r1c2"]


def test_missing_level_on_a_first_column_cell() -> None:
    proposal = propose(table_html(body='<tr><td data-level="0">a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr>'))
    assert codes(proposal.issues) == [CoreIssue.MISSING_LEVEL]
    issue = proposal.issues[0]
    assert (issue.blocking, issue.object_id) == (False, "r1c0")
    assert proposal.table is not None
    assert proposal.table.rows[1].cells[0].level is None


def test_level_outside_the_first_column_is_dropped() -> None:
    proposal = propose(
        table_html(body='<tr><td data-level="0">a</td><td>1</td><td data-level="1">b</td><td>2</td></tr>')
    )
    assert codes(proposal.issues) == [CoreIssue.UNEXPECTED_MARKUP]
    assert (proposal.issues[0].blocking, proposal.issues[0].object_id) == (False, "r0c2")
    assert proposal.table is not None
    assert proposal.table.rows[0].cells[2].level is None


def test_level_on_a_header_cell_is_dropped() -> None:
    proposal = propose(table_html(head='<tr><th data-level="0">A</th></tr>', body='<tr><td data-level="0">x</td></tr>'))
    assert codes(proposal.issues) == [CoreIssue.UNEXPECTED_MARKUP]
    assert proposal.issues[0].object_id == "h0"


def test_label_spanning_value_columns_is_flagged() -> None:
    proposal = propose(
        table_html(
            body='<tr><td data-level="0" colspan="2">Receipts</td></tr><tr><td data-level="1">a</td><td>1</td></tr>'
        )
    )
    assert codes(proposal.issues) == [CoreIssue.LABEL_SPANS_VALUES]
    assert (proposal.issues[0].blocking, proposal.issues[0].object_id) == (False, "r0c0")


def test_single_column_table_has_no_label_spanning_issue() -> None:
    assert propose(table_html(body='<tr><td data-level="0">only</td></tr>')).issues == ()


def test_empty_body_row_is_kept_without_cells() -> None:
    proposal = propose(table_html(body='<tr><td data-level="0">a</td></tr><tr></tr>'))
    assert proposal.table is not None
    assert proposal.table.rows[1] == Row(id="r1", cells=())


def test_table_without_any_cells_has_no_candidate() -> None:
    table = ParsedTable(
        fragment=2,
        title=None,
        head=(),
        body=(ParsedRow(source=SourceRef(fragment=2, section=Section.BODY, row=0), cells=()),),
        foot=(),
    )
    proposal = propose_table(table, table_id="t")
    assert proposal.table is None
    assert codes(proposal.issues) == [CoreIssue.MALFORMED_FRAGMENT]
    assert proposal.issues[0].source == SourceRef(fragment=2)


# Notes


def notes_for(foot: str, body: str = '<tr><td data-level="0">x</td><td>1</td></tr>') -> TableProposal:
    return propose(table_html(body=body, foot=foot))


@pytest.mark.parametrize(
    ("line", "label", "text"),
    [
        ("1/ Provisional figure.", "1/", "Provisional figure."),
        ("p/ Preliminary.", "p/", "Preliminary."),
        ("* Less than $500,000.", "*", "Less than $500,000."),
        ("†† Estimated.", "††", "Estimated."),
        ("Source: Daily Treasury Statement.", None, "Source: Daily Treasury Statement."),
        ("Note: Figures are rounded.", None, "Note: Figures are rounded."),
        ("p Preliminary.", None, "p Preliminary."),
        ("1234/ Too long.", None, "1234/ Too long."),
        ("1/", None, "1/"),
        ("  1/   Padded.  ", "1/", "Padded."),
    ],
)
def test_split_note_label(line: str, label: str | None, text: str) -> None:
    assert split_note_label(line) == (label, text)


@pytest.mark.parametrize(
    ("line", "markers", "expected"),
    [
        ("p Preliminary.", ["p"], ("p", "Preliminary.")),
        ("r Revised.", ["r/"], ("r", "Revised.")),
        ("12 Includes x.", ["12"], ("12", "Includes x.")),
        ("An analysis of the table.", ["p"], (None, "An analysis of the table.")),
        ("An analysis of the table.", ["Anx"], (None, "An analysis of the table.")),
        ("See footnotes to Table 1.", [], (None, "See footnotes to Table 1.")),
    ],
)
def test_bare_label_needs_a_matching_marker(line: str, markers: list[str], expected: tuple[str | None, str]) -> None:
    assert split_note_label(line, markers) == expected


@pytest.mark.parametrize(("label", "normalized"), [("1/", "1"), (" a/ ", "a"), ("1//", "1/"), ("*", "*"), ("p", "p")])
def test_normalize_note_label(label: str, normalized: str) -> None:
    assert normalize_note_label(label) == normalized


def test_superscript_label_at_the_start_of_a_note() -> None:
    proposal = notes_for(
        '<tr><td colspan="2"><sup>a</sup> Provisional.</td></tr>',
        body='<tr><td data-level="0">x</td><td>1<sup>a</sup></td></tr>',
    )
    assert proposal.issues == ()
    assert proposal.table is not None
    assert proposal.table.notes == (Note(id="n0", label="a", text="Provisional."),)
    assert proposal.table.rows[0].cells[1].markers == (Marker(at=1, text="a", note_ref="n0"),)


def test_bare_label_is_separated_when_the_table_has_that_marker() -> None:
    proposal = notes_for(
        '<tr><td colspan="2">p Preliminary.</td></tr><tr><td colspan="2">An analysis follows.</td></tr>',
        body='<tr><td data-level="0">x</td><td>1<sup>p</sup></td></tr>',
    )
    assert proposal.issues == ()
    assert proposal.table is not None
    assert proposal.table.notes == (
        Note(id="n0", label="p", text="Preliminary."),
        Note(id="n1", text="An analysis follows."),
    )


def test_footnote_row_with_several_cells_is_joined() -> None:
    proposal = notes_for("<tr><td>1/ Includes</td><td></td><td>Newfoundland.</td></tr>")
    assert codes(proposal.issues) == [CoreIssue.MALFORMED_FOOTNOTE_ROW]
    assert not proposal.issues[0].blocking
    assert proposal.table is not None
    assert proposal.table.notes == (Note(id="n0", label="1/", text="Includes Newfoundland."),)


def test_superscript_inside_a_note_is_kept_as_text() -> None:
    proposal = notes_for('<tr><td colspan="2">1/ Includes item<sup>2</sup> only.</td></tr>')
    assert codes(proposal.issues) == [CoreIssue.UNEXPECTED_MARKUP]
    assert proposal.issues[0].object_id == "n0"
    assert proposal.table is not None
    assert proposal.table.notes == (Note(id="n0", label="1/", text="Includes item2 only."),)


def test_empty_footnote_row_is_skipped() -> None:
    proposal = notes_for('<tr><td colspan="2"></td></tr><tr><td colspan="2">Source: X.</td></tr>')
    assert codes(proposal.issues) == [CoreIssue.UNEXPECTED_MARKUP]
    assert proposal.table is not None
    assert proposal.table.notes == (Note(id="n0", text="Source: X."),)
    assert proposal.sources["n0"] == SourceRef(fragment=0, section=Section.FOOT, row=1)


def test_unclear_footnote() -> None:
    proposal = notes_for('<tr><td colspan="2" data-unclear="1">Source: X.</td></tr>')
    assert proposal.table is not None
    assert proposal.table.notes == (Note(id="n0", text="Source: X.", unclear=True),)


def test_unlabeled_notes_are_not_attached() -> None:
    proposal = notes_for('<tr><td colspan="2">Source: X.</td></tr>')
    assert proposal.table is not None
    assert proposal.table.note_refs == ()


# Marker links


def test_unresolved_marker_points_at_its_cell() -> None:
    proposal = propose(table_html(body='<tr><td data-level="0">x<sup>3</sup></td><td>1</td></tr>'))
    assert codes(proposal.issues) == [CoreIssue.UNRESOLVED_MARKER]
    issue = proposal.issues[0]
    assert (issue.blocking, issue.object_id) == (False, "r0c0")
    assert issue.source == SourceRef(fragment=0, section=Section.BODY, row=0, cell=0)
    assert proposal.table is not None
    assert proposal.table.rows[0].cells[0].markers == (Marker(at=1, text="3"),)


def test_ambiguous_marker_stays_unlinked() -> None:
    proposal = notes_for(
        '<tr><td colspan="2">1/ First.</td></tr><tr><td colspan="2">1/ Second.</td></tr>',
        body='<tr><td data-level="0">x</td><td>1<sup>1</sup></td></tr>',
    )
    assert codes(proposal.issues) == [CoreIssue.AMBIGUOUS_NOTE_LABEL]
    assert "n0, n1" in proposal.issues[0].message
    assert proposal.table is not None
    assert proposal.table.rows[0].cells[1].markers == (Marker(at=1, text="1"),)


def test_title_marker_links_and_reports_against_the_title() -> None:
    linked = propose(
        table_html(
            title="Receipts<sup>1</sup>", body='<tr><td data-level="0">x</td></tr>', foot="<tr><td>1/ Note.</td></tr>"
        )
    )
    assert linked.issues == ()
    assert linked.table is not None
    assert linked.table.title == TextBlock(text="Receipts", markers=(Marker(at=8, text="1", note_ref="n0"),))

    unresolved = propose(table_html(title="Receipts<sup>2</sup>", body='<tr><td data-level="0">x</td></tr>'))
    assert codes(unresolved.issues) == [CoreIssue.UNRESOLVED_MARKER]
    assert unresolved.issues[0].object_id is None
    assert unresolved.issues[0].source == SourceRef(fragment=0, section=Section.TITLE)


def test_marker_with_slash_matches_label_without() -> None:
    proposal = notes_for(
        '<tr><td colspan="2"><sup>1</sup> Note.</td></tr>',
        body='<tr><td data-level="0">x</td><td>1<sup>1/</sup></td></tr>',
    )
    assert proposal.issues == ()
    assert proposal.table is not None
    assert proposal.table.rows[0].cells[1].markers[0].note_ref == "n0"
