"""Encoding and decoding table JSON."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from table_core.codec import TableDecodeError, decode_table, encode_table, schema_errors
from table_core.models import Cell, Column, Header, Marker, Row, Table

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = sorted((ROOT / "examples").glob("*.json"))


def load(name: str) -> Any:
    return json.loads((ROOT / "examples" / name).read_text())


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_round_trip_exactly(path: Path) -> None:
    # The examples are written in canonical form, so re-encoding reproduces them including key order.
    payload = json.loads(path.read_text())
    encoded = encode_table(decode_table(payload))
    assert json.dumps(encoded, ensure_ascii=False) == json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_decode_then_encode_is_identity_on_tables(path: Path) -> None:
    table = decode_table(json.loads(path.read_text()))
    assert decode_table(encode_table(table)) == table


def test_encode_omits_defaults_and_orders_fields() -> None:
    cell = Cell(id="a", columns=("c0",), text="10", markers=(Marker(at=2, text="1"),))
    table = Table(
        id="t",
        headers=(Header(id="h0", text="Item"),),
        columns=(Column(id="c0", headers=("h0",)),),
        rows=(Row(id="r0", cells=(cell,)),),
    )
    assert encode_table(table) == {
        "schema_version": "0.3",
        "id": "t",
        "headers": [{"id": "h0", "text": "Item"}],
        "columns": [{"id": "c0", "headers": ["h0"]}],
        "rows": [
            {"id": "r0", "cells": [{"id": "a", "columns": ["c0"], "text": "10", "markers": [{"at": 2, "text": "1"}]}]}
        ],
    }
    assert list(encode_table(table)) == ["schema_version", "id", "headers", "columns", "rows"]


def test_explicit_defaults_decode_to_canonical_form() -> None:
    payload = load("row-hierarchy.json")
    canonical = copy.deepcopy(payload)
    payload["label_columns"] = 1
    payload["notes"] = []
    payload["note_refs"] = []
    payload["title"]["unclear"] = False
    cell = payload["rows"][0]["cells"][1]
    cell.update({"rowspan": 1, "unclear": False, "markers": [], "note_refs": []})
    assert encode_table(decode_table(payload)) == canonical


def test_integral_float_is_decoded_as_int() -> None:
    payload = load("merged-cells.json")
    payload["rows"][0]["cells"][0]["rowspan"] = 2.0
    cell = decode_table(payload).rows[0].cells[0]
    assert cell.rowspan == 2
    assert isinstance(cell.rowspan, int)


def test_structural_problems_are_not_decode_errors() -> None:
    payload = load("row-hierarchy.json")
    payload["rows"][1]["cells"][1]["note_refs"] = ["no-such-note"]
    payload["rows"][2]["cells"] = []  # leaves positions uncovered
    decode_table(payload)


def _set(path: list[str | int], value: object) -> Callable[[Any], None]:
    def mutate(payload: Any) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _delete(path: list[str | int]) -> Callable[[Any], None]:
    def mutate(payload: Any) -> None:
        target = payload
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


INVALID = [
    pytest.param(_set(["schema_version"], "0.2"), "schema_version", id="unsupported version"),
    pytest.param(_delete(["id"]), "", id="missing table id"),
    pytest.param(_set(["id"], ""), "id", id="empty table id"),
    pytest.param(_set(["extra"], 1), "", id="unknown table property"),
    pytest.param(_set(["title"], None), "title", id="null instead of omitted"),
    pytest.param(_set(["columns"], []), "columns", id="no columns"),
    pytest.param(_set(["label_columns"], 0), "label_columns", id="zero label_columns"),
    pytest.param(_set(["columns", 0, "headers"], ["h0", "h0"]), "columns/0/headers", id="repeated column header"),
    pytest.param(_set(["rows", 0], []), "rows/0", id="row as bare array"),
    pytest.param(_delete(["rows", 0, "id"]), "rows/0", id="row without id"),
    pytest.param(_set(["rows", 0, "cells", 1, "rowspan"], 0), "rows/0/cells/1/rowspan", id="zero rowspan"),
    pytest.param(_set(["rows", 0, "cells", 1, "rowspan"], 1.5), "rows/0/cells/1/rowspan", id="fractional rowspan"),
    pytest.param(_set(["rows", 0, "cells", 0, "level"], -1), "rows/0/cells/0/level", id="negative level"),
    pytest.param(_set(["rows", 0, "cells", 0, "level"], True), "rows/0/cells/0/level", id="boolean level"),
    pytest.param(_set(["rows", 0, "cells", 0, "columns"], []), "rows/0/cells/0/columns", id="cell without columns"),
    pytest.param(
        _set(["rows", 0, "cells", 1, "markers"], [{"at": 0, "text": " "}]),
        "rows/0/cells/1/markers/0/text",
        id="whitespace marker text",
    ),
    pytest.param(
        _set(["rows", 0, "cells", 1, "markers"], [{"start": 0, "end": 1}]),
        "rows/0/cells/1/markers/0",
        id="0.2 marker range",
    ),
    pytest.param(_set(["rows", 0, "cells", 1, "note_refs"], ["n", "n"]), "rows/0/cells/1/note_refs", id="repeated ref"),
]


@pytest.mark.parametrize(("mutate", "path"), INVALID)
def test_invalid_payload_is_rejected(mutate: Callable[[Any], None], path: str) -> None:
    payload = load("row-hierarchy.json")
    mutate(payload)
    assert schema_errors(payload)
    with pytest.raises(TableDecodeError) as raised:
        decode_table(payload)
    assert raised.value.errors
    assert path in {e.path for e in raised.value.errors}


def test_non_object_payload_is_rejected() -> None:
    with pytest.raises(TableDecodeError):
        decode_table([])
