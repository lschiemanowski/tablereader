"""The JSON Schema shipped with table_core is itself valid, and every example table conforms to it."""

import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from table_core import SCHEMA_VERSION
from table_core.codec import SCHEMA_RESOURCE

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads(files("table_core").joinpath(SCHEMA_RESOURCE).read_text())
EXAMPLES = sorted((ROOT / "examples").glob("*.json"))


def test_schema_is_valid() -> None:
    Draft202012Validator.check_schema(SCHEMA)


def test_schema_version_matches_package() -> None:
    assert SCHEMA["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_examples_exist() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_conforms_to_schema(path: Path) -> None:
    errors = [e.message for e in Draft202012Validator(SCHEMA).iter_errors(json.loads(path.read_text()))]
    assert errors == []
