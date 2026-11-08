"""Cell grids from transcriptions, and their comparison with cell-grid references (table_benchmark v1).

v1 references are grids of a tight crop: header and body rows, no title or footnotes, markers flattened into
the text. To compare, a transcription's tables are laid out with the HTML table algorithm (head and body
stacked, footnote rows dropped, marker text put back) and several tables in one crop are stacked in reading
order. Texts are compared after `normalize_cell`, which also removes all whitespace and soft hyphens, so that
`Providers 32`, `Providers³²` and `Providers<sup>32</sup>` agree.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from table_core.parsing import ParsedRow, ParsedTable, ParsedText


@dataclass(frozen=True, slots=True)
class GridCell:
    row: int
    column: int
    row_span: int
    column_span: int
    text: str


@dataclass(frozen=True, slots=True)
class Grid:
    rows: int
    columns: int
    cells: tuple[GridCell, ...]


def normalize_cell(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).replace("\xad", "")
    s = re.sub(r"(?:[.·]\s*){3,}", "", s)  # dot leaders
    return "".join(s.casefold().split())


def _flat(text: ParsedText) -> str:
    pieces: list[str] = []
    pos = 0
    for marker in text.markers:
        pieces.extend((text.text[pos : marker.at], marker.text))
        pos = marker.at
    pieces.append(text.text[pos:])
    return "".join(pieces)


def _place(rows: Sequence[ParsedRow], offset: int) -> tuple[list[GridCell], int]:
    occupied: set[tuple[int, int]] = set()
    cells: list[GridCell] = []
    width = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row.cells:
            while (r, c) in occupied:
                c += 1
            rowspan = min(cell.rowspan, len(rows) - r)
            occupied.update((r + dr, c + dc) for dr in range(rowspan) for dc in range(cell.colspan))
            cells.append(GridCell(offset + r, c, rowspan, cell.colspan, _flat(cell.content)))
            c += cell.colspan
            width = max(width, c)
    return cells, width


def grid_from_parsed(tables: Iterable[ParsedTable]) -> Grid:
    """One grid of the head and body rows of `tables`, stacked in order."""
    cells: list[GridCell] = []
    rows = width = 0
    for table in tables:
        for section in (table.head, table.body):
            placed, section_width = _place(section, rows)
            cells.extend(placed)
            rows += len(section)
            width = max(width, section_width)
    return Grid(rows=rows, columns=width, cells=tuple(cells))


@dataclass(frozen=True, slots=True)
class GridComparison:
    """Agreement of a predicted grid with a reference grid; F1 scores over nonempty cells."""

    same_shape: bool
    structure_f1: float  # cell positions and spans
    text_f1: float  # positions and normalized text
    content_f1: float  # normalized texts as a multiset, ignoring positions
    reference_cells: int
    predicted_cells: int


def _f1(want: Counter[object], got: Counter[object]) -> float:
    total = sum(want.values()) + sum(got.values())
    return 2 * sum((want & got).values()) / total if total else 1.0


def compare_grids(reference: Grid, prediction: Grid) -> GridComparison:
    def nonempty(grid: Grid) -> list[tuple[GridCell, str]]:
        return [(c, t) for c in grid.cells if (t := normalize_cell(c.text))]

    want, got = nonempty(reference), nonempty(prediction)
    return GridComparison(
        same_shape=(reference.rows, reference.columns) == (prediction.rows, prediction.columns),
        structure_f1=_f1(
            Counter((c.row, c.column, c.row_span, c.column_span) for c, _ in want),
            Counter((c.row, c.column, c.row_span, c.column_span) for c, _ in got),
        ),
        text_f1=_f1(Counter((c.row, c.column, t) for c, t in want), Counter((c.row, c.column, t) for c, t in got)),
        content_f1=_f1(Counter(t for _, t in want), Counter(t for _, t in got)),
        reference_cells=len(want),
        predicted_cells=len(got),
    )
