"""Parsing model responses under the fixed transcription contract."""

import pytest

from table_core.issues import CoreIssue, Issue, Section, SourceRef
from table_core.parsing import (
    ParsedCell,
    ParsedFragment,
    ParsedMarker,
    ParsedOutput,
    ParsedTable,
    ParsedText,
    ResponseKind,
    parse_output,
)

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

SIMPLE = "<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>{}</td></tr></tbody></table>"


def codes(issues: tuple[Issue, ...]) -> list[str]:
    return [i.code for i in issues]


def only_table(output: ParsedOutput) -> ParsedTable:
    assert output.kind is ResponseKind.TABLES
    assert len(output.fragments) == 1
    table = output.fragments[0].table
    assert table is not None
    return table


def only_fragment(text: str) -> ParsedFragment:
    output = parse_output(text)
    assert len(output.fragments) == 1
    return output.fragments[0]


def cell(text: str) -> ParsedCell:
    """The single body cell of a one-cell table whose cell content is `text`."""
    return only_table(parse_output(SIMPLE.format(text))).body[0].cells[0]


# Response shape


def test_example_from_the_spec() -> None:
    output = parse_output(EXAMPLE)
    assert output.issues == ()
    assert output.fragments[0].issues == ()
    table = only_table(output)
    assert table.title == ParsedText(text="Receipts (in dollars)")
    assert [c.content.text for c in table.head[0].cells] == ["Item", "2025"]
    assert all(c.header for c in table.head[0].cells)
    receipts, taxes = table.body
    assert (receipts.cells[0].content.text, receipts.cells[0].level, receipts.cells[1].content.text) == (
        "Receipts:",
        0,
        "",
    )
    assert taxes.cells[1].content == ParsedText(text="100", markers=(ParsedMarker(at=3, text="1"),))
    footnote = table.foot[0].cells[0]
    assert (footnote.content.text, footnote.colspan) == ("1/ Provisional figure.", 2)


BODY_ONLY = """<table>
  <tbody>
    <tr><td data-level="0">Wheat</td><td>1,204</td><td>1,310</td></tr>
    <tr><td data-level="0">Barley</td><td>388</td><td>-</td></tr>
  </tbody>
</table>"""


def test_table_without_title_or_headings_from_the_spec() -> None:
    output = parse_output(BODY_ONLY)
    assert output.issues == ()
    assert output.fragments[0].issues == ()
    table = only_table(output)
    assert (table.title, table.head, table.foot) == (None, (), ())
    assert [[c.content.text for c in row.cells] for row in table.body] == [
        ["Wheat", "1,204", "1,310"],
        ["Barley", "388", "-"],
    ]


def test_source_refs_count_rows_and_cells_as_written() -> None:
    table = only_table(parse_output(EXAMPLE))
    assert table.body[1].source == SourceRef(fragment=0, section=Section.BODY, row=1)
    assert table.body[1].cells[1].source == SourceRef(fragment=0, section=Section.BODY, row=1, cell=1)
    assert table.foot[0].cells[0].source == SourceRef(fragment=0, section=Section.FOOT, row=0, cell=0)


@pytest.mark.parametrize("text", ["NOT_A_TABLE", "  NOT_A_TABLE\n"])
def test_sentinel(text: str) -> None:
    assert parse_output(text) == ParsedOutput(kind=ResponseKind.NOT_A_TABLE, fragments=())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", [CoreIssue.EMPTY_RESPONSE]),
        ("   \n", [CoreIssue.EMPTY_RESPONSE]),
        ("This page has no table.", [CoreIssue.UNEXPECTED_CONTENT]),
        ("NOT_A_TABLE\nThis is a contents page.", [CoreIssue.UNEXPECTED_CONTENT]),
        ("# Title only", [CoreIssue.UNEXPECTED_CONTENT]),
        ("| A | B |\n|---|---|\n| 1 | 2 |", [CoreIssue.PIPE_TABLE]),
    ],
)
def test_invalid_responses(text: str, expected: list[str]) -> None:
    output = parse_output(text)
    assert output.kind is ResponseKind.INVALID
    assert output.fragments == ()
    assert codes(output.issues) == expected
    assert all(i.blocking for i in output.issues)


def test_sentinel_with_tables_is_a_blocking_response_issue() -> None:
    output = parse_output("NOT_A_TABLE\n" + EXAMPLE)
    assert output.kind is ResponseKind.TABLES
    assert codes(output.issues) == [CoreIssue.MIXED_SENTINEL]


@pytest.mark.parametrize(
    "text",
    [
        "The details are shown in the following table.\n" + EXAMPLE,
        EXAMPLE + "\n1/ Includes gold released from earmark.",
        "```html\n" + EXAMPLE + "\n```",
        EXAMPLE + "\n# A heading with no table after it",
    ],
    ids=["prose before", "footnote after", "code fence", "trailing heading"],
)
def test_text_outside_tables_is_blocking_but_tables_are_kept(text: str) -> None:
    output = parse_output(text)
    assert codes(output.issues) == [CoreIssue.UNEXPECTED_CONTENT]
    assert output.issues[0].blocking
    only_table(output)


def test_several_tables_with_their_own_titles() -> None:
    text = "# First\n" + SIMPLE.format("1") + "\n\n" + SIMPLE.format("2") + "\n# Third\n" + SIMPLE.format("3")
    output = parse_output(text)
    titles = [f.table.title.text if f.table and f.table.title else None for f in output.fragments]
    assert titles == ["First", None, "Third"]  # a title is not inherited by later tables
    assert [f.fragment for f in output.fragments] == [0, 1, 2]
    assert output.issues == ()


def test_title_with_marker_and_entity() -> None:
    title = only_table(parse_output("# Yields &amp; Rates <sup>5</sup>\n" + SIMPLE.format("1"))).title
    assert title == ParsedText(text="Yields & Rates", markers=(ParsedMarker(at=14, text="5"),))


@pytest.mark.parametrize(
    ("heading", "title"),
    [
        ("## Table 1.- Treasury Bills", "Table 1.- Treasury Bills"),
        ("#Treasury Bills", "Treasury Bills"),
        ("# Section I - Securities\n## Table 1.- Treasury Bills", "Section I - Securities Table 1.- Treasury Bills"),
    ],
    ids=["level 2", "no space", "several lines"],
)
def test_irregular_titles_are_accepted_with_title_format(heading: str, title: str) -> None:
    fragment = only_fragment(heading + "\n" + SIMPLE.format("1"))
    assert fragment.table is not None and fragment.table.title == ParsedText(text=title)
    assert codes(fragment.issues) == [CoreIssue.TITLE_FORMAT]
    assert not fragment.issues[0].blocking
    assert fragment.issues[0].source == SourceRef(fragment=0, section=Section.TITLE)


def test_caption_is_blocking_and_kept_as_title_for_inspection() -> None:
    fragment = only_fragment("<table><caption>Receipts</caption><tbody><tr><td>1</td></tr></tbody></table>")
    assert codes(fragment.issues) == [CoreIssue.CAPTION_ELEMENT]
    assert fragment.table is not None and fragment.table.title == ParsedText(text="Receipts")


def test_unterminated_table_is_parsed_with_blocking_issue() -> None:
    fragment = only_fragment("<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2")
    assert codes(fragment.issues) == [CoreIssue.UNTERMINATED_TABLE]
    assert fragment.table is not None
    assert [r.cells[0].content.text for r in fragment.table.body] == ["1", "2"]


def test_nested_table_makes_the_fragment_unusable() -> None:
    fragment = only_fragment("<table><tbody><tr><td><table><tr><td>x</td></tr></table></td></tr></tbody></table>")
    assert fragment.table is None
    assert codes(fragment.issues) == [CoreIssue.NESTED_TABLE]


def test_empty_table_is_malformed() -> None:
    fragment = only_fragment("<table></table>")
    assert fragment.table is None
    assert codes(fragment.issues) == [CoreIssue.MALFORMED_FRAGMENT]


def test_broken_table_does_not_affect_its_neighbours() -> None:
    output = parse_output(SIMPLE.format("1") + "<table></table>" + SIMPLE.format("3"))
    assert [f.table is not None for f in output.fragments] == [True, False, True]
    assert output.issues == ()


def test_html_repair_is_recorded() -> None:
    fragment = only_fragment(SIMPLE.format("a</div>"))
    assert codes(fragment.issues) == [CoreIssue.HTML_REPAIRED]
    assert not fragment.issues[0].blocking


def test_implied_end_tags_are_not_repairs() -> None:
    fragment = only_fragment("<table><tbody><tr><td>a<td>b</tr></tbody></table>")
    assert fragment.issues == ()
    assert fragment.table is not None
    assert [c.content.text for c in fragment.table.body[0].cells] == ["a", "b"]


def test_text_between_rows_is_blocking() -> None:
    fragment = only_fragment("<table><tbody>stray<tr><td>1</td></tr></tbody></table>")
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_CONTENT]


# Text and markers


@pytest.mark.parametrize(
    ("html", "text", "markers"),
    [
        ("100<sup>1</sup>", "100", [(3, "1")]),
        (
            "Description of security <sup>2</sup> (Date of issue)",
            "Description of security (Date of issue)",
            [(23, "2")],
        ),
        ("<sup>a</sup> Total", "Total", [(0, "a")]),
        ("10<sup>1</sup><sup>2</sup>", "10", [(2, "1"), (2, "2")]),
        ("10<sup>*</sup> / 20<sup>*</sup>", "10 / 20", [(2, "*"), (7, "*")]),
        ("A😀<sup>1</sup>", "A😀", [(2, "1")]),
        ("  Total\n   receipts  ", "Total receipts", []),
        ("1,234&nbsp;5", "1,234\N{NO-BREAK SPACE}5", []),
        ("x &amp; y &lt;5", "x & y <5", []),
        ("Taxes <sup> 1 </sup>", "Taxes", [(5, "1")]),
        ("", "", []),
    ],
)
def test_inline_text_and_markers(html: str, text: str, markers: list[tuple[int, str]]) -> None:
    content = cell(html).content
    assert content.text == text
    assert [(m.at, m.text) for m in content.markers] == markers


def test_comments_are_ignored() -> None:
    assert cell("a<!-- note -->b").content.text == "ab"


@pytest.mark.parametrize(
    ("html", "text"),
    [("Total<br>receipts", "Total receipts"), ("<em>Total</em> receipts", "Total receipts")],
)
def test_inline_markup_keeps_text_with_non_blocking_issue(html: str, text: str) -> None:
    fragment = only_fragment(SIMPLE.format(html))
    assert fragment.table is not None
    assert fragment.table.body[0].cells[0].content.text == text
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_MARKUP]
    assert not fragment.issues[0].blocking


def test_empty_sup_is_not_a_marker() -> None:
    fragment = only_fragment(SIMPLE.format("10<sup> </sup>"))
    assert fragment.table is not None
    assert fragment.table.body[0].cells[0].content.markers == ()
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_MARKUP]


# Attributes


def body_cell_fragment(td: str) -> tuple[ParsedCell, list[str]]:
    fragment = only_fragment(f"<table><tbody><tr>{td}</tr></tbody></table>")
    assert fragment.table is not None
    return fragment.table.body[0].cells[0], codes(fragment.issues)


def test_spans_level_and_unclear() -> None:
    parsed, issues = body_cell_fragment('<td rowspan="2" colspan=" 3 " data-level="1" data-unclear="1">x</td>')
    assert (parsed.rowspan, parsed.colspan, parsed.level, parsed.unclear) == (2, 3, 1, True)
    assert issues == []


def test_missing_attributes_take_defaults() -> None:
    parsed, issues = body_cell_fragment("<td>x</td>")
    assert (parsed.rowspan, parsed.colspan, parsed.level, parsed.unclear) == (1, 1, None, False)
    assert issues == []


@pytest.mark.parametrize("value", ["0", "-1", "two", ""])
def test_invalid_span_is_blocking_and_defaults_to_one(value: str) -> None:
    parsed, issues = body_cell_fragment(f'<td rowspan="{value}">x</td>')
    assert parsed.rowspan == 1
    assert issues == [CoreIssue.INVALID_SPAN]


@pytest.mark.parametrize("value", ["-1", "one", "1.5"])
def test_invalid_level_is_dropped_without_blocking(value: str) -> None:
    parsed, issues = body_cell_fragment(f'<td data-level="{value}">x</td>')
    assert parsed.level is None
    assert issues == [CoreIssue.INVALID_LEVEL]


def test_other_unclear_value_still_counts_as_unclear() -> None:
    parsed, issues = body_cell_fragment('<td data-unclear="yes">x</td>')
    assert parsed.unclear
    assert issues == [CoreIssue.UNEXPECTED_MARKUP]


def test_unknown_attribute_is_ignored_with_issue() -> None:
    _, issues = body_cell_fragment('<td style="color: red">x</td>')
    assert issues == [CoreIssue.UNEXPECTED_MARKUP]


def test_one_markup_issue_per_cell() -> None:
    _, issues = body_cell_fragment('<td style="x" data-unclear="2">a<br>b</td>')
    assert issues == [CoreIssue.UNEXPECTED_MARKUP]


def test_th_in_body_is_read_as_cell() -> None:
    parsed, issues = body_cell_fragment("<th>Label</th>")
    assert parsed.header and parsed.content.text == "Label"
    assert issues == [CoreIssue.UNEXPECTED_MARKUP]


def test_row_outside_sections_is_a_body_row() -> None:
    fragment = only_fragment("<table><tr><td>a</td></tr></table>")
    assert fragment.table is not None
    assert [r.cells[0].content.text for r in fragment.table.body] == ["a"]
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_MARKUP]


def test_rows_outside_sections_are_reported_once() -> None:
    fragment = only_fragment("<table><tr><td>a</td></tr><tr><td>b</td></tr></table>")
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_MARKUP]


def test_element_with_text_between_cells_is_blocking() -> None:
    fragment = only_fragment("<table><tbody><tr><td>10</td><sup>1</sup><td>20</td></tr></tbody></table>")
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_CONTENT]
    assert fragment.issues[0].blocking


def test_empty_element_between_cells_is_not_blocking() -> None:
    fragment = only_fragment("<table><tbody><tr><td>10</td><span></span><td>20</td></tr></tbody></table>")
    assert codes(fragment.issues) == [CoreIssue.UNEXPECTED_MARKUP]
