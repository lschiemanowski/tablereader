"""Constructor shape checks and marker reinsertion."""

from collections.abc import Callable

import pytest

from table_core.models import Cell, Column, Header, Marker, Note, Row, Table, TextBlock, with_markers

COLUMN = Column(id="c0", headers=())


def test_valid_objects_construct() -> None:
    cell = Cell(id="a", columns=("c0",), text="10", rowspan=2, level=0, markers=(Marker(at=2, text="1"),))
    table = Table(id="t", headers=(), columns=(COLUMN,), rows=(Row(id="r0", cells=(cell,)), Row(id="r1", cells=())))
    assert table.rows[0].cells[0].rowspan == 2
    assert Cell(id="b", columns=("c0",), text="").level is None


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Marker(at=-1, text="1"), id="negative marker position"),
        pytest.param(lambda: Marker(at=0, text=""), id="empty marker text"),
        pytest.param(lambda: Marker(at=0, text=" \n"), id="whitespace marker text"),
        pytest.param(lambda: Marker(at=0, text="1", note_ref=""), id="empty marker note_ref"),
        pytest.param(lambda: TextBlock(text="T", note_refs=("n1", "n1")), id="repeated title note_ref"),
        pytest.param(lambda: Header(id="", text="H"), id="empty header id"),
        pytest.param(lambda: Header(id="h", text="H", note_refs=("",)), id="empty header note_ref"),
        pytest.param(lambda: Column(id="", headers=()), id="empty column id"),
        pytest.param(lambda: Column(id="c", headers=("h", "h")), id="repeated column header"),
        pytest.param(lambda: Cell(id="", columns=("c0",), text=""), id="empty cell id"),
        pytest.param(lambda: Cell(id="a", columns=(), text=""), id="cell without columns"),
        pytest.param(lambda: Cell(id="a", columns=("c0", "c0"), text=""), id="repeated cell column"),
        pytest.param(lambda: Cell(id="a", columns=("c0",), text="", rowspan=0), id="zero rowspan"),
        pytest.param(lambda: Cell(id="a", columns=("c0",), text="", level=-1), id="negative level"),
        pytest.param(lambda: Row(id="", cells=()), id="empty row id"),
        pytest.param(lambda: Note(id="", text="N"), id="empty note id"),
        pytest.param(lambda: Note(id="n", text="N", label=""), id="empty note label"),
        pytest.param(lambda: Table(id="", headers=(), columns=(COLUMN,), rows=()), id="empty table id"),
        pytest.param(lambda: Table(id="t", headers=(), columns=(), rows=()), id="table without columns"),
        pytest.param(
            lambda: Table(id="t", headers=(), columns=(COLUMN,), rows=(), label_columns=0), id="zero label_columns"
        ),
    ],
)
def test_constructor_rejects_bad_shape(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()


def test_structurally_invalid_table_constructs() -> None:
    # Dangling references and uncovered positions are validation issues, not constructor errors.
    cell = Cell(id="a", columns=("missing",), text="", note_refs=("n9",))
    Table(id="t", headers=(), columns=(COLUMN, Column(id="c1", headers=("h9",))), rows=(Row(id="r0", cells=(cell,)),))


@pytest.mark.parametrize(
    ("text", "markers", "expected"),
    [
        ("10", (Marker(at=2, text="1"),), "101"),
        ("101", (), "101"),
        ("Not disclosed", (Marker(at=13, text="*"),), "Not disclosed*"),
        ("10 / 20", (Marker(at=2, text="*"), Marker(at=7, text="*")), "10* / 20*"),
        ("10", (Marker(at=2, text="1"), Marker(at=2, text="2")), "1012"),
        ("A😀", (Marker(at=2, text="1"),), "A😀1"),
        ("Taxes", (Marker(at=0, text="a"),), "aTaxes"),
    ],
)
def test_with_markers(text: str, markers: tuple[Marker, ...], expected: str) -> None:
    assert with_markers(text, markers) == expected


@pytest.mark.parametrize(
    "markers",
    [(Marker(at=3, text="1"),), (Marker(at=2, text="1"), Marker(at=1, text="2"))],
    ids=["outside text", "decreasing"],
)
def test_with_markers_rejects_bad_positions(markers: tuple[Marker, ...]) -> None:
    with pytest.raises(ValueError):
        with_markers("10", markers)
