"""Propose table JSON from a parsed table.

`propose_table` turns the syntactic `ParsedTable` into a `Table`: it expands header spans into shared header
objects and column paths, turns body rows into rows and cells, separates note labels, and links markers to
notes. It always returns a candidate when the parsed table has any usable structure, even if that candidate
is structurally invalid; `validate_table` decides that.

Cells are placed with the HTML table algorithm, separately within `thead` and `tbody`. Body rows are not
padded: an uncovered body position stays uncovered and validation reports it. The header grid is not part of
the JSON, so its gaps and overlaps are reported here, as blocking issues, because they make column paths
unreliable.

Object IDs are assigned deterministically from source position, so the same parsed table always yields the
same IDs: headers `h0, h1, ...` in the order written, columns `c0, ...`, rows `r0, ...`, cells `r<row>c<column>`
(row index and first covered column), notes `n0, ...`. IDs stay opaque to consumers.
"""

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace

from table_core.issues import CoreIssue, Issue, Section, SourceRef, Stage, core_issue
from table_core.models import Cell, Column, Header, Marker, Note, Row, Table, TextBlock, with_markers
from table_core.parsing import ParsedCell, ParsedMarker, ParsedRow, ParsedTable, ParsedText


@dataclass(frozen=True, slots=True, kw_only=True)
class TableProposal:
    """A candidate table (or None), the issues found while building it, and where each object came from.

    `sources` maps object IDs (headers, rows, cells, notes) to the transcription elements they were built
    from. Columns have no single source element and are not listed.
    """

    table: Table | None
    issues: tuple[Issue, ...] = ()
    sources: Mapping[str, SourceRef] = field(default_factory=dict[str, SourceRef])


def propose_table(parsed: ParsedTable, *, table_id: str) -> TableProposal:
    """Build a candidate `Table` from `parsed`. Never raises on bad content."""
    return _Proposer(parsed).build(table_id)


_SLASH_LABEL = re.compile(r"([A-Za-z0-9]{1,3}/|[*†‡§]+)\s+(\S.*)", re.DOTALL)
_BARE_LABEL = re.compile(r"([A-Za-z0-9]{1,3})\s+(\S.*)", re.DOTALL)


def split_note_label(line: str, marker_texts: Collection[str] = ()) -> tuple[str | None, str]:
    """Separate a printed note label from a footnote line, per the rule in docs/table-format.md.

    `split_note_label("1/ Provisional figure.") == ("1/", "Provisional figure.")`;
    `split_note_label("Source: Daily Treasury Statement.") == (None, "Source: Daily Treasury Statement.")`.
    A label without a slash (`p Preliminary.`) is separated only when it equals the normalized text of one of
    `marker_texts`, the markers found in the same table. A leading `<sup>` label arrives as a marker at
    position 0 and is handled by the caller.
    """
    line = line.strip()
    if match := _SLASH_LABEL.fullmatch(line):
        return match.group(1), match.group(2).strip()
    if (match := _BARE_LABEL.fullmatch(line)) and match.group(1) in {normalize_note_label(t) for t in marker_texts}:
        return match.group(1), match.group(2).strip()
    return None, line


def normalize_note_label(label: str) -> str:
    """The form used to match markers to notes: surrounding whitespace and one trailing slash removed."""
    label = label.strip()
    return label.removesuffix("/")


@dataclass(slots=True)
class _Placed:
    """A cell placed in a section grid: its first column, the columns it covers and its clipped rowspan."""

    cell: ParsedCell
    row: int
    column: int
    colspan: int
    rowspan: int


class _Proposer:
    def __init__(self, parsed: ParsedTable) -> None:
        self.parsed = parsed
        self.issues: list[Issue] = []
        self.sources: dict[str, SourceRef] = {}

    def issue(self, code: CoreIssue, message: str, *, object_id: str | None = None, source: SourceRef) -> None:
        self.issues.append(core_issue(code, message, Stage.STRUCTURE, object_id=object_id, source=source))

    def build(self, table_id: str) -> TableProposal:
        head = self.place(self.parsed.head)
        body = self.place(self.parsed.body)
        width = max((p.column + p.colspan for p in head + body), default=0)
        if width == 0:
            source = SourceRef(fragment=self.parsed.fragment)
            self.issue(CoreIssue.MALFORMED_FRAGMENT, "the table has no cells", source=source)
            return TableProposal(table=None, issues=tuple(self.issues))

        column_ids = tuple(f"c{c}" for c in range(width))
        headers, columns = self.header_paths(head, column_ids)
        rows = self.body_rows(body, column_ids)
        notes = self.notes()
        title, headers, rows = self.link_markers(notes, headers, rows)
        table = Table(id=table_id, title=title, headers=headers, columns=columns, rows=rows, notes=notes)
        return TableProposal(table=table, issues=tuple(self.issues), sources=self.sources)

    # Grid placement

    def place(self, rows: tuple[ParsedRow, ...]) -> list[_Placed]:
        """Place cells with the HTML table algorithm; rowspans are clipped to the section."""
        occupied: set[tuple[int, int]] = set()
        placed: list[_Placed] = []
        for r, row in enumerate(rows):
            c = 0
            for cell in row.cells:
                while (r, c) in occupied:
                    c += 1
                rowspan = min(cell.rowspan, len(rows) - r)
                if rowspan < cell.rowspan:
                    self.issue(
                        CoreIssue.UNEXPECTED_MARKUP,
                        f"rowspan {cell.rowspan} reaches past the end of the section; shortened to {rowspan}",
                        source=cell.source,
                    )
                placed.append(_Placed(cell=cell, row=r, column=c, colspan=cell.colspan, rowspan=rowspan))
                occupied.update((r + dr, c + dc) for dr in range(rowspan) for dc in range(cell.colspan))
                c += cell.colspan
        return placed

    def header_paths(
        self, head: list[_Placed], column_ids: tuple[str, ...]
    ) -> tuple[tuple[Header, ...], tuple[Column, ...]]:
        """Header objects in the order written, and each column's path through the header grid."""
        width = len(column_ids)
        grid: dict[tuple[int, int], str] = {}
        headers: list[Header] = []
        for index, p in enumerate(head):
            header_id = f"h{index}"
            if p.cell.level is not None:
                self.issue(
                    CoreIssue.UNEXPECTED_MARKUP,
                    "data-level on a header cell is dropped",
                    object_id=header_id,
                    source=p.cell.source,
                )
            content = p.cell.content
            headers.append(
                Header(id=header_id, text=content.text, markers=_markers(content.markers), unclear=p.cell.unclear)
            )
            self.sources[header_id] = p.cell.source
            overlapped = False
            for r in range(p.row, p.row + p.rowspan):
                for c in range(p.column, p.column + p.colspan):
                    if (r, c) in grid:
                        overlapped = True
                    else:
                        grid[(r, c)] = header_id
            if overlapped:
                self.issue(
                    CoreIssue.OVERLAPPING_CELLS,
                    "header cell overlaps a cell placed earlier; the earlier cell keeps the shared positions",
                    object_id=header_id,
                    source=p.cell.source,
                )

        for r, row in enumerate(self.parsed.head):
            gaps = [c for c in range(width) if (r, c) not in grid]
            if gaps:
                self.issue(
                    CoreIssue.UNCOVERED_POSITION,
                    f"header row {r} covers no cell in column(s) {', '.join(map(str, gaps))} of {width}; "
                    "the header spans do not add up",
                    source=row.source,
                )

        columns: list[Column] = []
        for c, column_id in enumerate(column_ids):
            path: list[str] = []
            for r in range(len(self.parsed.head)):
                header_id = grid.get((r, c))
                if header_id is not None and (not path or path[-1] != header_id):
                    path.append(header_id)
            columns.append(Column(id=column_id, headers=tuple(path)))
        return tuple(headers), tuple(columns)

    def body_rows(self, body: list[_Placed], column_ids: tuple[str, ...]) -> tuple[Row, ...]:
        by_row: list[list[Cell]] = [[] for _ in self.parsed.body]
        for p in sorted(body, key=lambda p: (p.row, p.column)):
            cell_id = f"r{p.row}c{p.column}"
            self.sources[cell_id] = p.cell.source
            level = p.cell.level
            if p.column == 0:
                if level is None:
                    self.issue(
                        CoreIssue.MISSING_LEVEL,
                        "first-column cell has no data-level",
                        object_id=cell_id,
                        source=p.cell.source,
                    )
                if p.colspan > 1 and len(column_ids) > 1:
                    self.issue(
                        CoreIssue.LABEL_SPANS_VALUES,
                        f"first-column cell spans {p.colspan} columns, into the value columns",
                        object_id=cell_id,
                        source=p.cell.source,
                    )
            elif level is not None:
                self.issue(
                    CoreIssue.UNEXPECTED_MARKUP,
                    f"data-level outside the first column (column {p.column}) is dropped",
                    object_id=cell_id,
                    source=p.cell.source,
                )
                level = None
            content = p.cell.content
            by_row[p.row].append(
                Cell(
                    id=cell_id,
                    columns=column_ids[p.column : p.column + p.colspan],
                    text=content.text,
                    rowspan=p.rowspan,
                    level=level,
                    unclear=p.cell.unclear,
                    markers=_markers(content.markers),
                )
            )
        rows: list[Row] = []
        for r, (parsed_row, cells) in enumerate(zip(self.parsed.body, by_row, strict=True)):
            row_id = f"r{r}"
            self.sources[row_id] = parsed_row.source
            rows.append(Row(id=row_id, cells=tuple(cells)))
        return tuple(rows)

    # Notes and links

    def marker_texts(self) -> list[str]:
        texts = [m.text for row in self.parsed.head + self.parsed.body for c in row.cells for m in c.content.markers]
        if self.parsed.title is not None:
            texts.extend(m.text for m in self.parsed.title.markers)
        return texts

    def notes(self) -> tuple[Note, ...]:
        marker_texts = self.marker_texts()
        notes: list[Note] = []
        for row in self.parsed.foot:
            if len(row.cells) > 1:
                self.issue(
                    CoreIssue.MALFORMED_FOOTNOTE_ROW,
                    f"footnote row has {len(row.cells)} cells; their texts are joined",
                    source=row.source,
                )
            content = _join([c.content for c in row.cells])
            if not content.text and not content.markers:
                self.issue(CoreIssue.UNEXPECTED_MARKUP, "empty footnote row is skipped", source=row.source)
                continue
            note_id = f"n{len(notes)}"
            markers = content.markers
            if markers and markers[0].at == 0 and content.text:
                label: str | None = markers[0].text.strip()
                text = with_markers(content.text, _markers(markers[1:]))
                inner = markers[1:]
            else:
                label, text = split_note_label(with_markers(content.text, _markers(markers)), marker_texts)
                inner = markers
            if inner:
                self.issue(
                    CoreIssue.UNEXPECTED_MARKUP,
                    "superscript inside a footnote is kept as text",
                    object_id=note_id,
                    source=row.source,
                )
            notes.append(Note(id=note_id, text=text, label=label, unclear=any(c.unclear for c in row.cells)))
            self.sources[note_id] = row.source
        return tuple(notes)

    def link_markers(
        self, notes: tuple[Note, ...], headers: tuple[Header, ...], rows: tuple[Row, ...]
    ) -> tuple[TextBlock | None, tuple[Header, ...], tuple[Row, ...]]:
        by_label: dict[str, list[str]] = {}
        for note in notes:
            if note.label is not None:
                by_label.setdefault(normalize_note_label(note.label), []).append(note.id)

        def link(markers: tuple[Marker, ...], object_id: str | None, source: SourceRef) -> tuple[Marker, ...]:
            linked: list[Marker] = []
            for marker in markers:
                matches = by_label.get(normalize_note_label(marker.text), [])
                if len(matches) == 1:
                    linked.append(Marker(at=marker.at, text=marker.text, note_ref=matches[0]))
                    continue
                if matches:
                    code = CoreIssue.AMBIGUOUS_NOTE_LABEL
                    message = f"marker {marker.text!r} matches {len(matches)} notes: {', '.join(matches)}"
                else:
                    code = CoreIssue.UNRESOLVED_MARKER
                    message = f"marker {marker.text!r} matches no note label"
                self.issue(code, message, object_id=object_id, source=source)
                linked.append(marker)
            return tuple(linked)

        title = None
        if self.parsed.title is not None:
            source = SourceRef(fragment=self.parsed.fragment, section=Section.TITLE)
            markers = link(_markers(self.parsed.title.markers), None, source)
            title = TextBlock(text=self.parsed.title.text, markers=markers)
        new_headers = tuple(
            replace(h, markers=link(h.markers, h.id, self.sources[h.id])) if h.markers else h for h in headers
        )
        new_rows = tuple(
            Row(
                id=row.id,
                cells=tuple(
                    replace(c, markers=link(c.markers, c.id, self.sources[c.id])) if c.markers else c for c in row.cells
                ),
            )
            for row in rows
        )
        return title, new_headers, new_rows


def _markers(parsed: Sequence[ParsedMarker]) -> tuple[Marker, ...]:
    return tuple(Marker(at=m.at, text=m.text) for m in parsed)


def _join(texts: Sequence[ParsedText]) -> ParsedText:
    """Texts joined with single spaces, marker positions shifted accordingly. Empty texts add no space."""
    text = ""
    markers: list[ParsedMarker] = []
    for part in texts:
        if part.text and text:
            text += " "
        markers.extend(ParsedMarker(at=m.at + len(text), text=m.text) for m in part.markers)
        text += part.text
    return ParsedText(text=text, markers=tuple(markers))
