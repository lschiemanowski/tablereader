"""Parse the fixed OCR transcription (docs/transcription-format.md) into a syntactic representation.

`parse_output` only reads what was written: response shape, fragments and titles, rows and cells as
written (spans not expanded), attributes, text and `<sup>` markers. It does not decide what a header path,
a heading row, or a note label is; `structure.propose_table` does that.

Text in this representation is already normalized as the table format requires: HTML character references
decoded, runs of HTML whitespace collapsed to one space, ends stripped, `<sup>` content removed into markers.
A marker's position is directly after the last non-whitespace character before it. No other Unicode
normalization is applied.

The response is split by scanning for `<table>...</table>` blocks and Markdown heading lines; each table is
parsed separately with lxml, so a broken table cannot damage its neighbours.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum

import lxml.html
from lxml import etree

from table_core.issues import CoreIssue, Issue, Section, SourceRef, Stage, core_issue

SENTINEL = "NOT_A_TABLE"


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedMarker:
    """`<sup>` content removed from the text, and the code-point position where it was."""

    at: int
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedText:
    """Normalized text of a title or cell, with its markers in printed order."""

    text: str
    markers: tuple[ParsedMarker, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedCell:
    """A `th` or `td` as written. Invalid spans and levels have already become issues and defaults."""

    source: SourceRef
    header: bool  # th rather than td
    content: ParsedText
    rowspan: int = 1
    colspan: int = 1
    level: int | None = None
    unclear: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedRow:
    """A `tr` as written."""

    source: SourceRef
    cells: tuple[ParsedCell, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedTable:
    """One `<table>` and its title, as written."""

    fragment: int  # zero-based index of the table in the response, in reading order
    title: ParsedText | None
    head: tuple[ParsedRow, ...]
    body: tuple[ParsedRow, ...]
    foot: tuple[ParsedRow, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedFragment:
    """One table fragment of a response: the table if it could be read, and issues found in it."""

    fragment: int
    table: ParsedTable | None
    issues: tuple[Issue, ...] = ()


class ResponseKind(StrEnum):
    """What a response as a whole contains."""

    TABLES = "tables"  # one or more table fragments (some may have failed to parse)
    NOT_A_TABLE = "not_a_table"  # exactly the sentinel
    INVALID = "invalid"  # nothing usable: empty, prose only, or another contract violation without fragments


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedOutput:
    """A parsed response: its kind, fragments in reading order, and response-level issues.

    Response-level blocking issues (for example a sentinel mixed with tables) apply to every fragment.
    """

    kind: ResponseKind
    fragments: tuple[ParsedFragment, ...]
    issues: tuple[Issue, ...] = ()


def parse_output(text: str) -> ParsedOutput:
    """Split a raw model response into fragments and parse each one.

    Never raises on bad content: contract violations become issues. A malformed fragment does not invalidate
    other fragments when the boundaries between them are unambiguous.
    """
    text = text.strip()
    if not text:
        return ParsedOutput(
            kind=ResponseKind.INVALID,
            fragments=(),
            issues=(_issue(CoreIssue.EMPTY_RESPONSE, "the response is empty"),),
        )
    if text == SENTINEL:
        return ParsedOutput(kind=ResponseKind.NOT_A_TABLE, fragments=())

    blocks = _find_tables(text)
    bounds = [0, *(x for b in blocks for x in (b.start, b.end)), len(text)]
    segments = [_read_segment(text[bounds[i] : bounds[i + 1]]) for i in range(0, len(bounds), 2)]
    # Headings after the last table introduce nothing; they are stray content.
    trailing = segments[-1]
    trailing.content[:0] = [f"# {h}" for h in trailing.headings]
    trailing.headings.clear()

    issues: list[Issue] = []
    content = [line for s in segments for line in s.content]
    if any(s.pipe for s in segments):
        issues.append(_issue(CoreIssue.PIPE_TABLE, "the response contains a Markdown pipe table"))
    if content:
        issues.append(
            _issue(
                CoreIssue.UNEXPECTED_CONTENT,
                f"{len(content)} line(s) of text outside tables, starting with {_quote(content[0])}",
            )
        )
    sentinel = any(s.sentinel for s in segments)

    if not blocks:
        if not issues:
            issues.append(_issue(CoreIssue.UNEXPECTED_CONTENT, "the response contains no table"))
        return ParsedOutput(kind=ResponseKind.INVALID, fragments=(), issues=tuple(issues))
    if sentinel:
        issues.append(_issue(CoreIssue.MIXED_SENTINEL, f"{SENTINEL} appears together with tables"))
    fragments = tuple(
        _parse_fragment(i, text[b.start : b.end], b, segments[i].headings, segments[i].title_format)
        for i, b in enumerate(blocks)
    )
    return ParsedOutput(kind=ResponseKind.TABLES, fragments=fragments, issues=tuple(issues))


# Splitting the response


_TABLE_TAG = re.compile(r"<(/?)table\b[^>]*>", re.IGNORECASE)
_HEADING = re.compile(r"(#+)(\s*)(.*)")
_PIPE_ROW = re.compile(r"\|.*\|")


@dataclass(slots=True)
class _Block:
    start: int
    end: int
    terminated: bool
    nested: bool


@dataclass(slots=True)
class _Segment:
    """Text between tables: heading lines (titles of the next table) and anything else."""

    headings: list[str] = field(default_factory=list[str])
    title_format: bool = False  # a heading level other than 1 or a missing space after '#'
    content: list[str] = field(default_factory=list[str])
    pipe: bool = False
    sentinel: bool = False


def _find_tables(text: str) -> list[_Block]:
    """Top-level `<table>` blocks, tracking nesting; an unclosed table runs to the end of the text."""
    blocks: list[_Block] = []
    depth, start, nested = 0, 0, False
    for m in _TABLE_TAG.finditer(text):
        if m.group(1) != "/":
            if depth == 0:
                start, nested = m.start(), False
            else:
                nested = True
            depth += 1
        elif depth > 0:  # a stray </table> at depth 0 stays in the surrounding text
            depth -= 1
            if depth == 0:
                blocks.append(_Block(start, m.end(), terminated=True, nested=nested))
    if depth > 0:
        blocks.append(_Block(start, len(text), terminated=False, nested=nested))
    return blocks


def _read_segment(text: str) -> _Segment:
    segment = _Segment()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == SENTINEL:
            segment.sentinel = True
        elif heading := _HEADING.fullmatch(stripped):
            hashes, space, title = heading.groups()
            if len(hashes) != 1 or not space:
                segment.title_format = True
            segment.headings.append(title.strip())
        elif _PIPE_ROW.fullmatch(stripped):
            segment.pipe = True
        else:
            segment.content.append(stripped)
    return segment


# Parsing one table


def _parse_fragment(index: int, html: str, block: _Block, headings: list[str], title_format: bool) -> ParsedFragment:
    issues: list[Issue] = []
    title_ref = SourceRef(fragment=index, section=Section.TITLE)
    fragment_ref = SourceRef(fragment=index)
    title = None
    nonempty = [h for h in headings if h]
    if title_format or len(headings) > 1 or len(nonempty) < len(headings):
        issues.append(_issue(CoreIssue.TITLE_FORMAT, "the title is not a single '# ' heading line", title_ref))
    if nonempty:
        title = _parse_inline_html(" ".join(nonempty), issues, title_ref)
    if not block.terminated:
        issues.append(_issue(CoreIssue.UNTERMINATED_TABLE, "the table is not closed", fragment_ref))
    if block.nested:
        issues.append(_issue(CoreIssue.NESTED_TABLE, "the table contains another table", fragment_ref))
        return ParsedFragment(fragment=index, table=None, issues=tuple(issues))

    parser = lxml.html.HTMLParser(recover=True)
    try:
        root = lxml.html.fragment_fromstring(html, parser=parser)
    except (etree.ParserError, ValueError) as error:
        issues.append(_issue(CoreIssue.MALFORMED_FRAGMENT, f"the table could not be parsed: {error}", fragment_ref))
        return ParsedFragment(fragment=index, table=None, issues=tuple(issues))
    if len(parser.error_log):
        messages = "; ".join(e.message for e in parser.error_log)
        issues.append(_issue(CoreIssue.HTML_REPAIRED, f"the HTML parser repaired the markup: {messages}", fragment_ref))

    reader = _TableReader(index, issues)
    reader.read(root)
    if title is None and reader.caption is not None and reader.caption.text:
        title = reader.caption  # kept for inspection; the caption itself is a blocking contract issue
    if not (reader.head or reader.body or reader.foot):
        issues.append(_issue(CoreIssue.MALFORMED_FRAGMENT, "the table has no rows", fragment_ref))
        return ParsedFragment(fragment=index, table=None, issues=tuple(issues))
    table = ParsedTable(
        fragment=index, title=title, head=tuple(reader.head), body=tuple(reader.body), foot=tuple(reader.foot)
    )
    return ParsedFragment(fragment=index, table=table, issues=tuple(issues))


_SECTIONS = {"thead": Section.HEAD, "tbody": Section.BODY, "tfoot": Section.FOOT}
_CELL_ATTRIBUTES = frozenset({"rowspan", "colspan", "data-level", "data-unclear"})
_UNSIGNED = re.compile(r"[0-9]+")


class _TableReader:
    """Reads the sections, rows and cells of one `<table>` element, collecting issues."""

    def __init__(self, fragment: int, issues: list[Issue]) -> None:
        self.fragment = fragment
        self.issues = issues
        self.head: list[ParsedRow] = []
        self.body: list[ParsedRow] = []
        self.foot: list[ParsedRow] = []
        self.caption: ParsedText | None = None

    def rows(self, section: Section) -> list[ParsedRow]:
        return {Section.HEAD: self.head, Section.BODY: self.body, Section.FOOT: self.foot}[section]

    def read(self, root: lxml.html.HtmlElement) -> None:
        ref = SourceRef(fragment=self.fragment)
        self.check_stray_text(root, ref)
        loose_rows = False
        for child in _elements(root):
            tag = child.tag
            if tag in _SECTIONS:
                self.read_section(child, _SECTIONS[tag])
            elif tag == "tr":
                if not loose_rows:
                    self.issues.append(_issue(CoreIssue.UNEXPECTED_MARKUP, "<tr> outside thead/tbody/tfoot", ref))
                    loose_rows = True
                self.read_row(child, Section.BODY)
            elif tag == "caption":
                self.issues.append(_issue(CoreIssue.CAPTION_ELEMENT, "the table has a <caption> element", ref))
                self.caption = _parse_inline(
                    child, self.issues, SourceRef(fragment=self.fragment, section=Section.TITLE)
                )
            else:
                self.ignore(child, "<table>", ref)

    def read_section(self, element: lxml.html.HtmlElement, section: Section) -> None:
        ref = SourceRef(fragment=self.fragment, section=section)
        self.check_stray_text(element, ref)
        for child in _elements(element):
            if child.tag == "tr":
                self.read_row(child, section)
            else:
                self.ignore(child, "a section", ref)

    def read_row(self, element: lxml.html.HtmlElement, section: Section) -> None:
        rows = self.rows(section)
        ref = SourceRef(fragment=self.fragment, section=section, row=len(rows))
        self.check_stray_text(element, ref)
        cells: list[ParsedCell] = []
        for child in _elements(element):
            if child.tag in ("td", "th"):
                cell_ref = SourceRef(fragment=self.fragment, section=section, row=len(rows), cell=len(cells))
                cells.append(self.read_cell(child, section, cell_ref))
            else:
                self.ignore(child, "a row", ref)
        rows.append(ParsedRow(source=ref, cells=tuple(cells)))

    def read_cell(self, element: lxml.html.HtmlElement, section: Section, ref: SourceRef) -> ParsedCell:
        unexpected: set[str] = set()
        header = element.tag == "th"
        if header != (section == Section.HEAD):
            unexpected.add(f"<{element.tag}> in {section.value}")
        unknown = sorted(set(element.attrib) - _CELL_ATTRIBUTES)
        unexpected.update(f"attribute {name}" for name in unknown)

        level = None
        raw_level = element.get("data-level")
        if raw_level is not None:
            if _UNSIGNED.fullmatch(raw_level.strip()):
                level = int(raw_level)
            else:
                self.issues.append(_issue(CoreIssue.INVALID_LEVEL, f"data-level={raw_level!r} was dropped", ref))
        raw_unclear = element.get("data-unclear")
        if raw_unclear is not None and raw_unclear != "1":
            unexpected.add(f"data-unclear={raw_unclear!r}")

        content = _parse_inline(element, self.issues, ref, unexpected)
        return ParsedCell(
            source=ref,
            header=header,
            content=content,
            rowspan=self.span(element, "rowspan", ref),
            colspan=self.span(element, "colspan", ref),
            level=level,
            unclear=raw_unclear is not None,
        )

    def span(self, element: lxml.html.HtmlElement, name: str, ref: SourceRef) -> int:
        raw = element.get(name)
        if raw is None:
            return 1
        if _UNSIGNED.fullmatch(raw.strip()) and int(raw) > 0:
            return int(raw)
        self.issues.append(_issue(CoreIssue.INVALID_SPAN, f"{name}={raw!r} is not a positive integer", ref))
        return 1

    def ignore(self, element: lxml.html.HtmlElement, where: str, ref: SourceRef) -> None:
        """An element outside the table structure. Ignoring it loses its text, so text there is blocking."""
        text = _collapse(element.text_content())
        if text:
            message = f"<{element.tag}> in {where} holds text outside cells: {_quote(text)}"
            self.issues.append(_issue(CoreIssue.UNEXPECTED_CONTENT, message, ref))
        else:
            self.issues.append(
                _issue(CoreIssue.UNEXPECTED_MARKUP, f"empty <{element.tag}> in {where} was ignored", ref)
            )

    def check_stray_text(self, element: lxml.html.HtmlElement, ref: SourceRef) -> None:
        """Text directly inside table structure (not in a cell) would be lost; it is blocking content."""
        pieces = [element.text or "", *(child.tail or "" for child in element)]
        stray = _collapse("".join(pieces))
        if stray:
            self.issues.append(
                _issue(CoreIssue.UNEXPECTED_CONTENT, f"text outside cells: {_quote(stray)}", ref),
            )


# Inline text: whitespace collapsing and <sup> markers


_HTML_WHITESPACE = " \t\n\f\r"
_WHITESPACE_RUN = re.compile(r"[ \t\n\f\r]+")


class _TextBuilder:
    """Accumulates text with collapsed whitespace and records markers after the last non-space character."""

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.markers: list[ParsedMarker] = []
        self.pending_space = False

    def add_text(self, text: str) -> None:
        for char in text:
            if char in _HTML_WHITESPACE:
                self.pending_space = True
                continue
            if self.pending_space and self.chars:
                self.chars.append(" ")
            self.pending_space = False
            self.chars.append(char)

    def add_marker(self, text: str) -> None:
        self.markers.append(ParsedMarker(at=len(self.chars), text=text))

    def result(self) -> ParsedText:
        return ParsedText(text="".join(self.chars), markers=tuple(self.markers))


def _parse_inline(
    element: lxml.html.HtmlElement, issues: list[Issue], ref: SourceRef, unexpected: set[str] | None = None
) -> ParsedText:
    """The normalized text and markers of an element's content. Adds one `unexpected_markup` issue if needed."""
    found = set[str]() if unexpected is None else unexpected
    builder = _TextBuilder()
    _walk_inline(element, builder, found)
    if found:
        issues.append(_issue(CoreIssue.UNEXPECTED_MARKUP, f"unexpected markup: {', '.join(sorted(found))}", ref))
    return builder.result()


def _parse_inline_html(html: str, issues: list[Issue], ref: SourceRef) -> ParsedText:
    """Like `_parse_inline`, for a string of inline HTML such as a title line."""
    element = lxml.html.fragment_fromstring(html, create_parent="span", parser=lxml.html.HTMLParser(recover=True))
    return _parse_inline(element, issues, ref)


def _walk_inline(element: lxml.html.HtmlElement, builder: _TextBuilder, unexpected: set[str]) -> None:
    builder.add_text(element.text or "")
    for child in element:
        if isinstance(child, lxml.html.HtmlElement):
            if child.tag == "sup":
                marker = _collapse(child.text_content())
                if marker:
                    builder.add_marker(marker)
                else:
                    unexpected.add("empty <sup>")
                if len(child):
                    unexpected.add("markup inside <sup>")
            elif child.tag == "br":
                unexpected.add("<br>")
                builder.add_text(" ")
            else:
                unexpected.add(f"<{child.tag}>")
                _walk_inline(child, builder, unexpected)
        builder.add_text(child.tail or "")


def _elements(element: lxml.html.HtmlElement) -> list[lxml.html.HtmlElement]:
    """Child elements, skipping comments and processing instructions."""
    return [child for child in element if isinstance(child, lxml.html.HtmlElement)]


def _collapse(text: str) -> str:
    return _WHITESPACE_RUN.sub(" ", text).strip(_HTML_WHITESPACE)


def _quote(text: str, limit: int = 80) -> str:
    return repr(text if len(text) <= limit else text[: limit - 1] + "…")


def _issue(code: CoreIssue, message: str, source: SourceRef | None = None) -> Issue:
    return core_issue(code, message, Stage.PARSE, source=source)
