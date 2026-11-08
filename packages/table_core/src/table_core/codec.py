"""Encode and decode table JSON (draft 0.3).

The JSON Schema shipped with this package (`table.schema.json`) is the authority on shape. `decode_table`
validates every payload against it before building objects, so the decoder itself contains no shape rules and
cannot drift from the schema.

`encode_table` produces the canonical form: `schema_version` first, fields in spec order, optional fields
omitted when they equal their default.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, relevance

from table_core import SCHEMA_VERSION
from table_core.models import Cell, Column, Header, Marker, Note, Row, Table, TextBlock

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]

SCHEMA_RESOURCE = "table.schema.json"


@dataclass(frozen=True, slots=True)
class SchemaError:
    """One schema violation: where it is (e.g. `rows/3/cells/0/rowspan`) and what is wrong."""

    path: str
    message: str


class TableDecodeError(ValueError):
    """The payload does not conform to the table JSON Schema.

    `errors` lists every violation the schema reports, most relevant first; the exception message shows the
    first one.
    """

    def __init__(self, errors: tuple[SchemaError, ...]) -> None:
        first = errors[0]
        super().__init__(f"{first.path}: {first.message}" if first.path else first.message)
        self.errors = errors


@cache
def table_schema() -> JsonObject:
    """The table JSON Schema, loaded once from the package resource `table.schema.json`."""
    return cast(JsonObject, json.loads(files("table_core").joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")))


@cache
def _validator() -> Draft202012Validator:
    return Draft202012Validator(table_schema())


def schema_errors(payload: JsonValue) -> tuple[SchemaError, ...]:
    """Every violation of the table JSON Schema in `payload`, most relevant first. Empty means conforming."""
    # The stubs declare a deprecated second iter_errors overload with an untyped parameter.
    found: list[ValidationError] = list(_validator().iter_errors(payload))  # pyright: ignore[reportUnknownMemberType]
    errors = sorted(found, key=relevance, reverse=True)
    return tuple(SchemaError("/".join(str(p) for p in e.absolute_path), e.message) for e in errors)


# Encoding. Each function writes fields in spec order and omits optional fields that hold their default.


def _encode_markers(out: JsonObject, markers: Sequence[Marker]) -> None:
    if markers:
        encoded: list[JsonValue] = []
        for m in markers:
            item: JsonObject = {"at": m.at, "text": m.text}
            if m.note_ref is not None:
                item["note_ref"] = m.note_ref
            encoded.append(item)
        out["markers"] = encoded


def _encode_note_refs(out: JsonObject, note_refs: Sequence[str]) -> None:
    if note_refs:
        out["note_refs"] = list[JsonValue](note_refs)


def _encode_title(title: TextBlock) -> JsonObject:
    out: JsonObject = {"text": title.text}
    _encode_markers(out, title.markers)
    _encode_note_refs(out, title.note_refs)
    if title.unclear:
        out["unclear"] = True
    return out


def _encode_header(header: Header) -> JsonObject:
    out: JsonObject = {"id": header.id, "text": header.text}
    _encode_markers(out, header.markers)
    _encode_note_refs(out, header.note_refs)
    if header.unclear:
        out["unclear"] = True
    return out


def _encode_column(column: Column) -> JsonObject:
    return {"id": column.id, "headers": list[JsonValue](column.headers)}


def _encode_cell(cell: Cell) -> JsonObject:
    out: JsonObject = {"id": cell.id, "columns": list[JsonValue](cell.columns), "text": cell.text}
    if cell.rowspan != 1:
        out["rowspan"] = cell.rowspan
    if cell.level is not None:
        out["level"] = cell.level
    if cell.unclear:
        out["unclear"] = True
    _encode_markers(out, cell.markers)
    _encode_note_refs(out, cell.note_refs)
    return out


def _encode_row(row: Row) -> JsonObject:
    return {"id": row.id, "cells": [_encode_cell(c) for c in row.cells]}


def _encode_note(note: Note) -> JsonObject:
    out: JsonObject = {"id": note.id}
    if note.label is not None:
        out["label"] = note.label
    out["text"] = note.text
    if note.unclear:
        out["unclear"] = True
    return out


def encode_table(table: Table) -> JsonObject:
    """The canonical JSON object for `table`. Does not validate structure; call `validate_table` for that."""
    out: JsonObject = {"schema_version": SCHEMA_VERSION, "id": table.id}
    if table.title is not None:
        out["title"] = _encode_title(table.title)
    if table.label_columns != 1:
        out["label_columns"] = table.label_columns
    out["headers"] = [_encode_header(h) for h in table.headers]
    out["columns"] = [_encode_column(c) for c in table.columns]
    out["rows"] = [_encode_row(r) for r in table.rows]
    if table.notes:
        out["notes"] = [_encode_note(n) for n in table.notes]
    _encode_note_refs(out, table.note_refs)
    return out


# Decoding. The payload has passed schema validation, so these accessors only narrow types.


def _obj(value: JsonValue) -> Mapping[str, JsonValue]:
    return cast(Mapping[str, JsonValue], value)


def _objs(o: Mapping[str, JsonValue], key: str) -> list[Mapping[str, JsonValue]]:
    return [_obj(v) for v in cast(list[JsonValue], o.get(key, []))]


def _str(o: Mapping[str, JsonValue], key: str) -> str:
    return cast(str, o[key])


def _opt_str(o: Mapping[str, JsonValue], key: str) -> str | None:
    return cast(str | None, o.get(key))


def _strs(o: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    return tuple(cast(list[str], o.get(key, [])))


def _int(o: Mapping[str, JsonValue], key: str, default: int) -> int:
    # JSON Schema accepts 2.0 as an integer; store it as a Python int.
    return int(cast(int | float, o.get(key, default)))


def _opt_int(o: Mapping[str, JsonValue], key: str) -> int | None:
    value = cast(int | float | None, o.get(key))
    return None if value is None else int(value)


def _bool(o: Mapping[str, JsonValue], key: str) -> bool:
    return cast(bool, o.get(key, False))


def _decode_markers(o: Mapping[str, JsonValue]) -> tuple[Marker, ...]:
    return tuple(
        Marker(at=_int(m, "at", 0), text=_str(m, "text"), note_ref=_opt_str(m, "note_ref")) for m in _objs(o, "markers")
    )


def _decode_title(o: Mapping[str, JsonValue]) -> TextBlock:
    return TextBlock(
        text=_str(o, "text"), markers=_decode_markers(o), note_refs=_strs(o, "note_refs"), unclear=_bool(o, "unclear")
    )


def _decode_header(o: Mapping[str, JsonValue]) -> Header:
    return Header(
        id=_str(o, "id"),
        text=_str(o, "text"),
        markers=_decode_markers(o),
        note_refs=_strs(o, "note_refs"),
        unclear=_bool(o, "unclear"),
    )


def _decode_cell(o: Mapping[str, JsonValue]) -> Cell:
    return Cell(
        id=_str(o, "id"),
        columns=_strs(o, "columns"),
        text=_str(o, "text"),
        rowspan=_int(o, "rowspan", 1),
        level=_opt_int(o, "level"),
        unclear=_bool(o, "unclear"),
        markers=_decode_markers(o),
        note_refs=_strs(o, "note_refs"),
    )


def decode_table(payload: JsonValue) -> Table:
    """Validate `payload` against the schema and build a `Table`. Raises `TableDecodeError` if it does not conform.

    An unsupported `schema_version` is a schema violation. Structural problems (dangling references, uncovered
    grid positions) are not: the result can be passed to `validate_table`.
    """
    errors = schema_errors(payload)
    if errors:
        raise TableDecodeError(errors)
    o = _obj(payload)
    title = o.get("title")
    return Table(
        id=_str(o, "id"),
        title=None if title is None else _decode_title(_obj(title)),
        label_columns=_int(o, "label_columns", 1),
        headers=tuple(_decode_header(h) for h in _objs(o, "headers")),
        columns=tuple(Column(id=_str(c, "id"), headers=_strs(c, "headers")) for c in _objs(o, "columns")),
        rows=tuple(
            Row(id=_str(r, "id"), cells=tuple(_decode_cell(c) for c in _objs(r, "cells"))) for r in _objs(o, "rows")
        ),
        notes=tuple(
            Note(id=_str(n, "id"), text=_str(n, "text"), label=_opt_str(n, "label"), unclear=_bool(n, "unclear"))
            for n in _objs(o, "notes")
        ),
        note_refs=_strs(o, "note_refs"),
    )
