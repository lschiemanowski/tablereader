"""Column and row paths derived from a table (docs/table-format.md, "Deriving row paths").

Paths are derived, never stored. Evaluation builds its records from them; nothing here interprets values.
Both functions assume a table that passes `validate_table`; on an invalid table the result is unspecified but
they do not raise.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from table_core.models import Cell, Table


@dataclass(frozen=True, slots=True, kw_only=True)
class RowPath:
    """The path of one body row.

    `labels` are the open headings' labels followed by this row's label, each label contributing its texts in
    column order. It is None when the path cannot be resolved because it depends on a missing level.
    `heading` says whether the row is a heading row (a label and no value text).
    """

    row_id: str
    labels: tuple[str, ...] | None
    heading: bool


def column_paths(table: Table) -> Mapping[str, tuple[str, ...]]:
    """Header texts from outermost to innermost, keyed by column ID. Blank headers are skipped."""
    texts = {header.id: header.text for header in table.headers}
    return {
        column.id: tuple(text for header_id in column.headers if (text := texts.get(header_id, "")))
        for column in table.columns
    }


def row_paths(table: Table) -> tuple[RowPath, ...]:
    """The path of every body row, in row order, using the heading rule."""
    grid = _resolve(table)
    label_columns = min(table.label_columns, len(table.columns))
    open_headings: list[_Open] = []
    paths: list[RowPath] = []
    for r, row in enumerate(table.rows):
        covering = grid[r]
        label_cells = _distinct(covering[:label_columns])
        label = tuple(cell.text for cell in label_cells if cell.text)
        has_value = any(cell.text and cell not in label_cells for cell in covering[label_columns:] if cell)
        first = covering[0] if covering else None
        level = first.level if first is not None else None

        if not label and not has_value:  # entirely blank: leaves the open headings as they are
            resolved = not any(h.maybe_closed for h in open_headings)
            labels = _flatten(open_headings) if resolved else None
            paths.append(RowPath(row_id=row.id, labels=labels, heading=False))
            continue

        heading = not has_value
        kept: list[_Open] = []
        resolved = True
        for h in open_headings:
            keep = _keeps(h.level, level, heading=heading)
            if keep is _Keep.NO:
                continue
            if keep is _Keep.UNKNOWN:
                kept.append(_Open(labels=h.labels, level=h.level, maybe_closed=True))
                resolved = False
            else:
                kept.append(h)
                resolved = resolved and not h.maybe_closed
        labels = _flatten(kept) + label if resolved else None
        paths.append(RowPath(row_id=row.id, labels=labels, heading=heading))
        open_headings = kept
        if heading:
            open_headings.append(_Open(labels=label, level=level, maybe_closed=False))
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class _Open:
    """An open heading. `maybe_closed`: a row of unknown level may already have closed it."""

    labels: tuple[str, ...]
    level: int | None
    maybe_closed: bool


class _Keep(Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def _keeps(open_level: int | None, level: int | None, *, heading: bool) -> _Keep:
    """Whether a row at `level` keeps an open heading at `open_level` open.

    A new heading keeps headings with a lower level; a value row keeps headings with a lower or equal level.
    Levels are nonnegative, which settles two cases with a missing level: a level-0 heading closes everything,
    and a value row of unknown level keeps a level-0 heading.
    """
    if open_level is not None and level is not None:
        keep = open_level < level if heading else open_level <= level
        return _Keep.YES if keep else _Keep.NO
    if heading and level == 0:
        return _Keep.NO
    if not heading and level is None and open_level == 0:
        return _Keep.YES
    return _Keep.UNKNOWN


def _resolve(table: Table) -> list[list[Cell | None]]:
    """For each row, the cell covering each column (None where uncovered)."""
    index = {column.id: i for i, column in enumerate(table.columns)}
    grid: list[list[Cell | None]] = [[None] * len(table.columns) for _ in table.rows]
    for r, row in enumerate(table.rows):
        for cell in row.cells:
            for covered in grid[r : r + cell.rowspan]:
                for column_id in cell.columns:
                    if column_id in index:
                        covered[index[column_id]] = cell
    return grid


def _distinct(cells: list[Cell | None]) -> list[Cell]:
    """Cells in column order, each once."""
    seen: list[Cell] = []
    for cell in cells:
        if cell is not None and all(cell is not s for s in seen):
            seen.append(cell)
    return seen


def _flatten(headings: list[_Open]) -> tuple[str, ...]:
    return tuple(text for h in headings for text in h.labels)
