"""Structural validation of a table (docs/table-format.md, "Validation and publication").

`validate_table` runs the structural checks that JSON Schema cannot express: ID uniqueness, reference
integrity, adjacent columns under a shared header, cell order, complete non-overlapping grid coverage, level
placement, `label_columns` bounds, marker positions, and Unicode scalar values. Every issue it returns is
blocking. Passing establishes structural consistency, not transcription accuracy.

Local shape (nonempty IDs, `rowspan >= 1`, nonnegative levels and positions, ...) is guaranteed by the model
constructors and not checked again. Issues come in the order of the spec's checklist and in table order within
each check. A broken reference or duplicate ID is reported once; later checks ignore the broken reference and
use the first object with a duplicated ID, so one fault does not cascade.
"""

from collections.abc import Iterable, Sequence

from table_core.issues import CoreIssue, Issue, Stage, core_issue
from table_core.models import Cell, Marker, Table


def validate_table(table: Table) -> tuple[Issue, ...]:
    """All structural problems in `table`, in a deterministic order. Empty means structurally valid."""
    return _Validator(table).run()


class _Validator:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.issues: list[Issue] = []
        self.header_ids: set[str] = set()
        self.column_index: dict[str, int] = {}
        self.note_ids: set[str] = set()

    def issue(self, code: CoreIssue, message: str, object_id: str | None = None) -> None:
        self.issues.append(core_issue(code, message, Stage.VALIDATION, object_id=object_id))

    def run(self) -> tuple[Issue, ...]:
        self.ids()
        self.references()
        self.headers()
        self.cells()
        self.levels()
        self.markers()
        self.note_refs()
        self.unicode()
        return tuple(self.issues)

    # 1. IDs and references

    def ids(self) -> None:
        t = self.table
        seen: dict[str, str] = {}
        objects: list[tuple[str, str]] = [
            *((h.id, "header") for h in t.headers),
            *((c.id, "column") for c in t.columns),
            *((r.id, "row") for r in t.rows),
            *((c.id, "cell") for r in t.rows for c in r.cells),
            *((n.id, "note") for n in t.notes),
        ]
        for object_id, kind in objects:
            if object_id in seen:
                self.issue(
                    CoreIssue.DUPLICATE_ID, f"{kind} ID {object_id!r} is already used by a {seen[object_id]}", object_id
                )
            else:
                seen[object_id] = kind
        self.header_ids = {h.id for h in t.headers}
        for i, column in enumerate(t.columns):
            self.column_index.setdefault(column.id, i)
        self.note_ids = {n.id for n in t.notes}

    def references(self) -> None:
        t = self.table
        for column in t.columns:
            self.dangling(column.id, repr(column.id), "header", column.headers, self.header_ids)
        for row in t.rows:
            for cell in row.cells:
                self.dangling(cell.id, repr(cell.id), "column", cell.columns, self.column_index)
        for object_id, markers, note_refs in self.text_objects():
            where = repr(object_id) if object_id is not None else "the title"
            marker_refs = [m.note_ref for m in markers if m.note_ref is not None]
            self.dangling(object_id, where, "note", marker_refs, self.note_ids)
            self.dangling(object_id, where, "note", note_refs, self.note_ids)
        self.dangling(None, "the table", "note", t.note_refs, self.note_ids)

    def dangling(self, owner: str | None, where: str, kind: str, refs: Iterable[str], known: Iterable[str]) -> None:
        known = set(known)
        for ref in refs:
            if ref not in known:
                self.issue(
                    CoreIssue.DANGLING_REFERENCE, f"{where} refers to {kind} {ref!r}, which does not exist", owner
                )

    def text_objects(self) -> list[tuple[str | None, tuple[Marker, ...], tuple[str, ...]]]:
        """(object ID, markers, note_refs) of the title (ID None), headers and cells, in table order."""
        t = self.table
        objects: list[tuple[str | None, tuple[Marker, ...], tuple[str, ...]]] = []
        if t.title is not None:
            objects.append((None, t.title.markers, t.title.note_refs))
        objects.extend((h.id, h.markers, h.note_refs) for h in t.headers)
        objects.extend((c.id, c.markers, c.note_refs) for r in t.rows for c in r.cells)
        return objects

    # 2. Header paths

    def headers(self) -> None:
        """Every header is referenced, and the columns sharing a header are adjacent.

        Headers need not form a tree: a units line printed under the headings across the full width sits under
        different headers in different columns.
        """
        columns: dict[str, list[int]] = {}
        for i, column in enumerate(self.table.columns):
            for header_id in column.headers:
                if header_id in self.header_ids:
                    columns.setdefault(header_id, []).append(i)
        for header in self.table.headers:
            indexes = columns.get(header.id)
            if indexes is None:
                self.issue(CoreIssue.UNREFERENCED_HEADER, f"header {header.id!r} is in no column path", header.id)
            elif indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                self.issue(
                    CoreIssue.HEADER_STRUCTURE,
                    f"header {header.id!r} is shared by columns that are not adjacent",
                    header.id,
                )

    # 3. Cells and grid coverage

    def cells(self) -> None:
        t = self.table
        width = len(t.columns)
        covered: list[list[bool]] = [[False] * width for _ in t.rows]
        unknown: set[int] = set()  # rows spanned by a cell with a dangling column: their gaps are not reported
        for r, row in enumerate(t.rows):
            previous: int | None = None
            for cell in row.cells:
                indexes = [self.column_index[c] for c in cell.columns if c in self.column_index]
                if len(indexes) < len(cell.columns):
                    unknown.update(range(r, r + cell.rowspan))
                if not indexes:
                    continue  # every column dangling: already reported
                if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                    self.issue(
                        CoreIssue.NONCONSECUTIVE_COLUMNS,
                        f"cell {cell.id!r} covers columns that are not consecutive and in order",
                        cell.id,
                    )
                first = min(indexes)
                if previous is not None and first < previous:
                    self.issue(
                        CoreIssue.CELL_ORDER,
                        f"cell {cell.id!r} starts left of the cell before it in row {row.id!r}",
                        cell.id,
                    )
                previous = first
                if r + cell.rowspan > len(t.rows):
                    self.issue(
                        CoreIssue.ROWSPAN_OUT_OF_BOUNDS,
                        f"cell {cell.id!r} spans {cell.rowspan} rows from row {row.id!r}, past the last row",
                        cell.id,
                    )
                if self.cover(covered, r, cell, indexes):
                    self.issue(
                        CoreIssue.OVERLAPPING_CELLS, f"cell {cell.id!r} overlaps a cell placed before it", cell.id
                    )
        for r, (row, flags) in enumerate(zip(t.rows, covered, strict=True)):
            gaps = [t.columns[i].id for i, flag in enumerate(flags) if not flag]
            if gaps and r not in unknown:
                self.issue(
                    CoreIssue.UNCOVERED_POSITION,
                    f"row {row.id!r}: no cell covers {', '.join(gaps)}",
                    row.id,
                )

    @staticmethod
    def cover(covered: list[list[bool]], r: int, cell: Cell, indexes: Sequence[int]) -> bool:
        """Mark the cell's positions (within bounds) as covered; True if any was covered already."""
        overlap = False
        for flags in covered[r : r + cell.rowspan]:
            for i in indexes:
                overlap = overlap or flags[i]
                flags[i] = True
        return overlap

    # 4. Levels and label columns

    def levels(self) -> None:
        t = self.table
        first = t.columns[0].id
        for row in t.rows:
            for cell in row.cells:
                if cell.level is not None and first not in cell.columns:
                    self.issue(
                        CoreIssue.MISPLACED_LEVEL,
                        f"cell {cell.id!r} has a level but does not cover the first column",
                        cell.id,
                    )
        width = len(t.columns)
        if t.label_columns > width or (width > 1 and t.label_columns == width):
            self.issue(
                CoreIssue.LABEL_COLUMNS_OUT_OF_RANGE,
                f"label_columns is {t.label_columns} in a table with {width} column(s); at least one value column "
                "must remain",
            )

    # 5. Marker positions

    def markers(self) -> None:
        t = self.table
        texts: list[tuple[str | None, str, tuple[Marker, ...]]] = []
        if t.title is not None:
            texts.append((None, t.title.text, t.title.markers))
        texts.extend((h.id, h.text, h.markers) for h in t.headers)
        texts.extend((c.id, c.text, c.markers) for r in t.rows for c in r.cells)
        for object_id, text, markers in texts:
            previous = 0
            for marker in markers:
                if marker.at > len(text) or marker.at < previous:
                    where = f"{object_id!r}" if object_id is not None else "the title"
                    self.issue(
                        CoreIssue.INVALID_MARKER_POSITION,
                        f"marker {marker.text!r} in {where} is at {marker.at}, outside the text (length {len(text)}) "
                        f"or before the marker preceding it (at {previous})",
                        object_id,
                    )
                previous = max(previous, marker.at)

    # 6. note_refs

    def note_refs(self) -> None:
        for object_id, markers, note_refs in self.text_objects():
            linked = {m.note_ref for m in markers if m.note_ref is not None}
            for ref in note_refs:
                if ref in linked:
                    where = f"{object_id!r}" if object_id is not None else "the title"
                    self.issue(
                        CoreIssue.DUPLICATE_NOTE_REF,
                        f"{where} lists note {ref!r} in note_refs although one of its markers links to it",
                        object_id,
                    )

    # 7. Unicode

    def unicode(self) -> None:
        t = self.table
        strings: list[tuple[str | None, list[str]]] = [(None, [t.id, *t.note_refs])]
        if t.title is not None:
            strings.append((None, [t.title.text, *_marker_strings(t.title.markers), *t.title.note_refs]))
        strings.extend((h.id, [h.id, h.text, *_marker_strings(h.markers), *h.note_refs]) for h in t.headers)
        strings.extend((c.id, [c.id, *c.headers]) for c in t.columns)
        for row in t.rows:
            strings.append((row.id, [row.id]))
            strings.extend(
                (c.id, [c.id, *c.columns, c.text, *_marker_strings(c.markers), *c.note_refs]) for c in row.cells
            )
        strings.extend((n.id, [n.id, n.text, *([n.label] if n.label is not None else [])]) for n in t.notes)
        for object_id, values in strings:
            if any(_has_surrogate(v) for v in values):
                where = f"{object_id!r}" if object_id is not None else "the table or its title"
                self.issue(CoreIssue.INVALID_UNICODE, f"{where} contains an unpaired surrogate", object_id)


def _marker_strings(markers: Iterable[Marker]) -> list[str]:
    return [s for m in markers for s in (m.text, m.note_ref) if s is not None]


def _has_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)
