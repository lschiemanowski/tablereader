"""The record metric (planning/architecture.md, "Evaluation"), computed from table JSON.

A record is one nonempty value cell as `(row path, column path, value text)`. Row paths follow the heading rule
(`table_core.paths`); where a missing level leaves a path unresolved, rows fall back to level 0, as ttr did, and
the number of such rows is reported. Column paths skip blank headers. A value spanning several columns or rows
is one record, at its starting row and first column. Labels are normalized (`normalize_label`); values are
compared as transcribed, markers already removed. A crop's score is |reference ∩ prediction| / |union| over the
records of all its tables; a crop both sides call NOT_A_TABLE scores 1, a disagreement 0, and a failed
prediction 0.

Metric version `records-1`. Change the version with any change to normalization or record construction.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from table_core.models import Row, Table
from table_core.paths import column_paths, row_paths

RECORDS_VERSION = "records-1"

Record = tuple[tuple[str, ...], tuple[str, ...], str]


class NotATable:
    """A crop judged to contain no table (the NOT_A_TABLE sentinel)."""

    def __repr__(self) -> str:
        return "NOT_A_TABLE"


NOT_A_TABLE = NotATable()
CropTables = Sequence[Table] | NotATable | None  # None: extraction failed


def normalize_label(text: str) -> str:
    """Row and column labels for matching: dot leaders, dashes, fractions, trailing punctuation, case, spacing.

    Ported from ttr's `norm_label` without its footnote-marker stripping: markers are separated from text now.
    """
    s = unicodedata.normalize("NFC", text)
    s = re.sub(r"\.{2,}|…", " ", s)
    s = re.sub("[\N{EN DASH}\N{EM DASH}]", "-", s).replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    s = re.sub(r"[.:,]+(\s|$)", r"\1", s).strip()
    s = re.sub(r"\s*-\s*", "-", s)
    return " ".join(s.casefold().split())


@dataclass(frozen=True, slots=True)
class TableRecords:
    records: frozenset[Record]
    level_fallback_rows: int  # rows whose path needed the level-0 fallback
    duplicates: int  # records produced more than once (hidden by the set)


def table_records(table: Table) -> TableRecords:
    resolved = row_paths(table)
    fallback_rows = sum(p.labels is None for p in resolved)
    if fallback_rows:
        resolved = row_paths(_levels_default_to_zero(table))
    paths = column_paths(table)
    column_index = {c.id: i for i, c in enumerate(table.columns)}
    first_value_column = min(table.label_columns, len(table.columns) - 1) if len(table.columns) > 1 else 0
    records: list[Record] = []
    for row, path in zip(table.rows, resolved, strict=True):
        labels = tuple(normalize_label(t) for t in (path.labels or ()))
        for cell in row.cells:
            first = column_index.get(cell.columns[0])
            if first is None or first < first_value_column or not cell.text:
                continue
            column = tuple(normalize_label(t) for t in paths[cell.columns[0]])
            records.append((labels, column, cell.text))
    unique = frozenset(records)
    return TableRecords(records=unique, level_fallback_rows=fallback_rows, duplicates=len(records) - len(unique))


def _levels_default_to_zero(table: Table) -> Table:
    first = table.columns[0].id
    rows = tuple(
        Row(
            id=row.id,
            cells=tuple(replace(c, level=0) if c.level is None and first in c.columns else c for c in row.cells),
        )
        for row in table.rows
    )
    return replace(table, rows=rows)


def crop_records(tables: Sequence[Table]) -> frozenset[Record]:
    found: set[Record] = set()
    for table in tables:
        found |= table_records(table).records
    return frozenset(found)


def crop_score(reference: CropTables, prediction: CropTables) -> float:
    """Record overlap for one crop, with the non-table and failure rules of the module docstring."""
    if reference is None:
        raise ValueError("a reference cannot be a failed extraction")
    if isinstance(reference, NotATable) or isinstance(prediction, NotATable):
        return 1.0 if isinstance(reference, NotATable) and isinstance(prediction, NotATable) else 0.0
    if prediction is None:
        return 0.0
    want, got = crop_records(reference), crop_records(prediction)
    union = want | got
    return len(want & got) / len(union) if union else 1.0
