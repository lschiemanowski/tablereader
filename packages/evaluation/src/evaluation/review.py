"""LLM review of first-pass transcriptions (planning/architecture.md, "LLM review"), evaluated before serving.

A reviewer sees the crop, a view of the draft (per table: title, numbered header rows, a body grid of numbered
rows and grid columns, footnote lines) and the parser's issues. It answers with a verdict and a list of edits, not
a new transcription. The edits it is offered (`STRUCTURE_OPS`) restructure the draft without retyping what the OCR
model read: replace the header or the title, shift the rest of a row by some columns, move a body cell's content
to another column of its row, merge a
wrapped label's two rows, move rows between header and body, delete a row. Body values are never retyped; a trial
showed a reviewer reading digits worse than the OCR model (planning/r2-notes.md). Earlier edit kinds
(`set_cell`, `set_level`, `insert_row`) are still applied so that stored answers can be rescored.

`judge` applies the edits to the draft's HTML and keeps the revision only when

- every edit's guard holds: its address exists, its `old` matches the draft, a moved cell lands on an empty cell
  (or one that moves away in the same answer), and its `new` parses as the element it replaces;
- it changes at most `max_changed_share` of the draft's body cells (a larger change is a rewrite, not a review);
- the revision parses into as many tables as the draft, with fewer blocking issues if the draft has any (none
  otherwise), and every table is still ready if every draft table was.

Blocking issues must fall because a revision that leaves them in place leaves the table unresolved, and on the
pilot reviewers such revisions were where the regressions were (planning/r2-notes.md, second pilot).

Otherwise the draft is kept and the outcome says why. This deviates from the architecture's reviewer result in
one respect: the complete revised candidate is built here from the edits, never written by the reviewer.
"""

import base64
import hashlib
import io
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from importlib.resources import files
from typing import Any, Protocol, cast

import httpx
import lxml.html
from lxml.html import HtmlElement
from PIL import Image

from extraction.extract import interpret
from extraction.results import ComponentFailure, OcrResult, RegionResult, RegionStatus
from extraction.vllm import model_image

PROMPT: str = files("evaluation").joinpath("prompts/review.md").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


class EditOp(StrEnum):
    SET_CELL = "set_cell"
    SET_LEVEL = "set_level"
    REPLACE_HEADER = "replace_header"
    SET_TITLE = "set_title"
    INSERT_ROW = "insert_row"
    DELETE_ROW = "delete_row"
    MOVE_CELL = "move_cell"
    SHIFT_CELLS = "shift_cells"
    MERGE_ROWS = "merge_rows"
    ROW_TO_HEADER = "row_to_header"
    HEADER_TO_ROW = "header_to_row"


STRUCTURE_OPS = (
    EditOp.REPLACE_HEADER, EditOp.SET_TITLE, EditOp.SHIFT_CELLS, EditOp.MOVE_CELL, EditOp.MERGE_ROWS,
    EditOp.ROW_TO_HEADER, EditOp.HEADER_TO_ROW, EditOp.DELETE_ROW,
)  # fmt: skip
"""The edits a reviewer is offered: none of them retypes a body value."""


class Verdict(StrEnum):
    UNCHANGED = "unchanged"
    REVISED = "revised"
    UNRESOLVED = "unresolved"


_NULLABLE_INT = {"type": ["integer", "null"]}
_NULLABLE_STR = {"type": ["string", "null"]}
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": [op.value for op in STRUCTURE_OPS]},
                    "table": {"type": "integer"},
                    "row": _NULLABLE_INT,
                    "col": _NULLABLE_INT,
                    "to_col": _NULLABLE_INT,
                    "old": _NULLABLE_STR,
                    "new": _NULLABLE_STR,
                    "reason": {"type": "string"},
                },
                "required": ["op", "table", "row", "col", "to_col", "old", "new", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "edits"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Edit:
    op: EditOp
    table: int
    row: int | None = None
    col: int | None = None
    old: str | None = None
    new: str | None = None
    reason: str = ""
    to_col: int | None = None  # move_cell, shift_cells: the column the content moves to

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Edit":
        return cls(
            op=EditOp(data["op"]), table=data["table"], row=data.get("row"), col=data.get("col"),
            old=data.get("old"), new=data.get("new"), reason=data.get("reason", ""), to_col=data.get("to_col"),
        )  # fmt: skip


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    region_id: str
    image: Image.Image
    draft: str  # the first-pass transcription


@dataclass(frozen=True, slots=True)
class ReviewAnswer:
    verdict: Verdict
    edits: tuple[Edit, ...]
    model: str
    settings: dict[str, Any] = field(default_factory=dict[str, Any])
    usage: dict[str, int] = field(default_factory=dict[str, int])
    cost: float | None = None  # as the provider reports it, in dollars
    raw: str = ""
    reasoning: str = ""  # the model's reasoning text, where the provider returns it


class TableReviewer(Protocol):
    async def review(self, request: ReviewRequest) -> ReviewAnswer: ...


class OutcomeState(StrEnum):
    UNCHANGED = "unchanged"  # the reviewer kept the draft
    REVISED = "revised"  # edits applied and accepted
    UNRESOLVED = "unresolved"  # the reviewer could not fix the draft; the draft is kept
    REJECTED = "rejected"  # the edits failed a check; the draft is kept


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    state: OutcomeState
    text: str  # the transcription to use: the revision when REVISED, the draft otherwise
    reason: str = ""
    changed_cells: int = 0


class EditError(ValueError):
    """An edit whose guard fails or that cannot be applied."""


# --- the draft as HTML elements ------------------------------------------------------------------------------

_TABLE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_SPACE = re.compile(r"\s+")


@dataclass(slots=True)
class _Table:
    title: str | None
    element: HtmlElement


@dataclass(frozen=True, slots=True)
class _Slot:
    cell: HtmlElement
    origin: bool  # the grid position where `cell` starts


def _norm(text: str | None) -> str:
    return _SPACE.sub(" ", text or "").strip()


def _span(cell: HtmlElement, name: str) -> int:
    value = cell.get(name, "1").strip()
    return int(value) if value.isdigit() and int(value) > 0 else 1


def inner_html(element: HtmlElement) -> str:
    """An element's content in transcription syntax: its text and child elements, without its own tag."""
    parts = [element.text or ""]
    parts += [lxml.html.tostring(child, encoding="unicode", with_tail=True) for child in element]
    return _norm("".join(parts))


def split_draft(text: str) -> list[_Table] | None:
    """The draft's tables with their titles, or None if it is not exactly a set of complete tables with titles
    (`NOT_A_TABLE`, an unterminated table, text outside tables): joining it again would silently drop text."""
    found = list(_TABLE.finditer(text))
    if not found or len(re.findall(r"<table\b", text, re.IGNORECASE)) != len(found):
        return None
    outside = [text[a.end() if a else 0 : b.start()] for a, b in zip([None, *found[:-1]], found, strict=True)]
    if text[found[-1].end() :].strip() or any(
        line.strip() and not line.strip().startswith("#") for gap in outside for line in gap.splitlines()
    ):
        return None
    tables: list[_Table] = []
    start = 0
    for match in found:
        headings = [line.strip() for line in text[start : match.start()].splitlines() if line.strip().startswith("#")]
        title = " ".join(h.lstrip("#").strip() for h in headings) or None
        tables.append(_Table(title, lxml.html.fragment_fromstring(match.group(0))))
        start = match.end()
    return tables


def join_draft(tables: Sequence[_Table]) -> str:
    parts: list[str] = []
    for table in tables:
        if table.title is not None:
            parts.append(f"# {table.title}")
        parts.append(lxml.html.tostring(table.element, encoding="unicode").strip())
    return "\n".join(parts)


def _body_rows(table: HtmlElement) -> list[HtmlElement]:
    return cast(list[HtmlElement], table.xpath("./tbody/tr | ./tr"))


def _cells(row: HtmlElement) -> list[HtmlElement]:
    return [c for c in row if isinstance(c.tag, str) and c.tag.lower() in ("td", "th")]


def _header_rows(table: HtmlElement) -> list[HtmlElement]:
    return cast(list[HtmlElement], table.xpath("./thead/tr"))


def body_grid(table: HtmlElement) -> list[dict[int, _Slot]]:
    """Body rows with spans expanded: per row, grid column -> the cell covering it."""
    return _grid(_body_rows(table))


def _grid(rows: Sequence[HtmlElement]) -> list[dict[int, _Slot]]:
    grid: list[dict[int, _Slot]] = []
    carry: dict[int, tuple[HtmlElement, int]] = {}  # column -> (cell, rows still covered)
    for row in rows:
        line = {col: _Slot(cell, False) for col, (cell, _) in carry.items()}
        carry = {col: (cell, left - 1) for col, (cell, left) in carry.items() if left > 1}
        col = 0
        for cell in _cells(row):
            while col in line:
                col += 1
            down, across = _span(cell, "rowspan"), _span(cell, "colspan")
            for k in range(across):
                line[col + k] = _Slot(cell, k == 0)
                if down > 1:
                    carry[col + k] = (cell, down - 1)
            col += across
        grid.append(line)
    return grid


def _parse_element(html: str, tag: str) -> HtmlElement:
    """`html` as one `<thead>` or `<tr>` element, parsed inside a table as the HTML parser expects."""
    try:
        holder = lxml.html.fragment_fromstring(f"<table>{html.strip()}</table>")
    except Exception as error:  # lxml raises ParserError and others for unusable fragments
        raise EditError(f"new is not a <{tag}> element: {error}") from error
    found = cast(list[HtmlElement], holder.xpath(f"./{tag} | ./tbody/{tag}"))
    if len(found) != 1 or len(holder) != 1:
        raise EditError(f"new is not exactly one <{tag}> element")
    return found[0]


def _as_thead(html: str) -> str:
    """`html` with bare header rows (`<tr>...</tr>` without the `<thead>` around them) wrapped in a `<thead>`."""
    stripped = html.strip()
    return f"<thead>{stripped}</thead>" if stripped.lower().startswith("<tr") else stripped


def _set_content(cell: HtmlElement, html: str) -> None:
    holder = lxml.html.fragment_fromstring(html, create_parent="td")
    for child in list(cell):
        cell.remove(child)
    cell.text = holder.text
    for child in list(holder):
        cell.append(child)


# --- the view the reviewer sees ------------------------------------------------------------------------------


def render_view(draft: str, issues: Sequence[str] = ()) -> str:
    """The draft as the reviewer sees it: per table, title, header HTML, numbered body grid and footnotes."""
    tables = split_draft(draft)
    if tables is None:
        raise ValueError("only drafts made of complete tables can be reviewed")
    out: list[str] = []
    for index, table in enumerate(tables):
        out.append(f"## Table {index}")
        out.append(f"Title: {json.dumps(table.title, ensure_ascii=False) if table.title is not None else 'none'}")
        heads = _header_rows(table.element)
        out.append("Header rows:" if heads else "Header: none")
        out += [
            f"h{i} {_norm(lxml.html.tostring(tr, encoding='unicode', with_tail=False))}" for i, tr in enumerate(heads)
        ]
        grid = body_grid(table.element)
        width = max((max(line) + 1 for line in grid if line), default=0)
        out.append(f"Body: {len(grid)} rows, {width} grid columns. ^ marks a position covered by a cell that")
        out.append("starts in an earlier row, < one covered by a cell that starts in an earlier column.")
        for r, line in enumerate(grid):
            first = line.get(0)
            level = first.cell.get("data-level") if first is not None and first.origin else None
            cells: list[str] = []
            for c in sorted(line):
                slot = line[c]
                if slot.origin:
                    cells.append(f"c{c} {json.dumps(inner_html(slot.cell), ensure_ascii=False)}")
                else:
                    above = c - 1 not in line or line[c - 1].cell is not slot.cell
                    cells.append(f"c{c} {'^' if above else '<'}")
            out.append(f"r{r} level={level if level is not None else 'none'} | " + " | ".join(cells))
        notes = cast(list[HtmlElement], table.element.xpath("./tfoot/tr"))
        if notes:
            out.append("Footnotes:")
            out += [json.dumps(inner_html(_cells(n)[0]) if _cells(n) else "", ensure_ascii=False) for n in notes]
        out.append("")
    out.append("Parser issues:")
    out += [f"- {issue}" for issue in issues] or ["none"]
    return "\n".join(out)


def draft_issues(region_id: str, draft: str) -> list[str]:
    result = _interpret(region_id, draft)
    found = [f"{'blocking' if i.blocking else 'note'}: {i.message}" for i in result.issues]
    for t in result.tables:
        found += [f"table {t.index}, {'blocking' if i.blocking else 'note'}: {i.message}" for i in t.issues]
    return list(dict.fromkeys(found))  # parse-level issues repeat in every table


# --- applying and judging edits ------------------------------------------------------------------------------


def apply_edits(draft: str, edits: Sequence[Edit]) -> tuple[str, int]:
    """The draft with `edits` applied, and the number of body cells they change. Raises EditError.

    Every address refers to the draft as given: all edits are resolved and checked before anything changes.
    """
    tables = split_draft(draft)
    if tables is None:
        raise EditError("the draft is not a set of complete tables")
    grids = [body_grid(t.element) for t in tables]
    rows = [_body_rows(t.element) for t in tables]
    widths = [_width(grid + _grid(_header_rows(t.element))) for t, grid in zip(tables, grids, strict=True)]
    actions: list[Callable[[], None]] = []
    moves: dict[tuple[int, int], list[Edit]] = defaultdict(list)
    to_header: dict[int, set[int]] = defaultdict(set)
    to_body: dict[int, set[int]] = defaultdict(set)
    touched: set[tuple[int, int]] = set()  # body rows an edit removes or rebuilds; each at most once
    headers: set[int] = set()  # tables whose header an edit replaces; each at most once
    changed = 0

    def touch(table: int, row: int) -> None:
        if (table, row) in touched:
            raise EditError(f"body row {row} is changed by more than one row edit")
        touched.add((table, row))

    for edit in edits:
        if not 0 <= edit.table < len(tables):
            raise EditError(f"no table {edit.table}")
        table, grid, body = tables[edit.table], grids[edit.table], rows[edit.table]
        match edit.op:
            case EditOp.SET_CELL:
                slot = _slot(grid, edit)
                if _norm(edit.old) != inner_html(slot.cell):
                    raise EditError(f"r{edit.row} c{edit.col} holds {inner_html(slot.cell)!r}, not {edit.old!r}")
                if edit.new is None:
                    raise EditError("set_cell needs new content")
                new = edit.new
                actions.append(lambda cell=slot.cell, new=new: _set_content(cell, new))
                changed += 1
            case EditOp.SET_LEVEL:
                slot = _slot(grid, Edit(EditOp.SET_LEVEL, edit.table, edit.row, 0))
                if _norm(edit.old) != _norm(slot.cell.get("data-level")):
                    raise EditError(f"r{edit.row} has level {slot.cell.get('data-level')!r}, not {edit.old!r}")
                if edit.new is None or not edit.new.strip().isdigit():
                    raise EditError(f"level {edit.new!r} is not a nonnegative integer")
                level = edit.new.strip()
                actions.append(lambda cell=slot.cell, level=level: cell.set("data-level", level))
            case EditOp.REPLACE_HEADER:
                if any(v is not None for v in (edit.row, edit.col, edit.to_col, edit.old)):
                    raise EditError("replace_header takes the whole header in new, no row, column or old value")
                if edit.table in headers:
                    raise EditError(f"table {edit.table} has more than one replace_header")
                headers.add(edit.table)
                header = _parse_element(_as_thead(edit.new), "thead") if edit.new is not None else None
                actions.append(lambda element=table.element, header=header: _replace_header(element, header))
            case EditOp.SET_TITLE:
                if _norm(edit.old) != _norm(table.title):
                    raise EditError(f"table {edit.table} has title {table.title!r}, not {edit.old!r}")
                title = _norm(edit.new) or None
                actions.append(lambda table=table, title=title: setattr(table, "title", title))
            case EditOp.INSERT_ROW:
                if edit.row is None or not 0 <= edit.row <= len(body):
                    raise EditError(f"cannot insert before body row {edit.row}")
                row = _parse_element(edit.new or "", "tr")
                anchor = body[edit.row] if edit.row < len(body) else None
                body_parent = body[-1].getparent() if body else None
                actions.append(lambda t=table.element, a=anchor, p=body_parent, r=row: _insert_row(t, a, p, r))
                changed += len(_cells(row))
            case EditOp.DELETE_ROW:
                label = _label(grid, edit)
                if _norm(edit.old) != label:
                    raise EditError(f"r{edit.row} starts with {label!r}, not {edit.old!r}")
                touch(edit.table, cast(int, edit.row))
                victim = body[cast(int, edit.row)]
                actions.append(lambda row=victim: _remove(row))
                changed += sum(slot.origin for slot in grid[cast(int, edit.row)].values())
            case EditOp.MOVE_CELL:
                if edit.row is None or not 0 <= edit.row < len(grid):
                    raise EditError(f"no body row {edit.row}")
                moves[(edit.table, edit.row)].append(edit)
                changed += 1
            case EditOp.SHIFT_CELLS:
                if edit.row is None or not 0 <= edit.row < len(grid):
                    raise EditError(f"no body row {edit.row}")
                touch(edit.table, edit.row)
                actions.append(_shift(grid[edit.row], body[edit.row], edit, widths[edit.table]))
                changed += 1
            case EditOp.MERGE_ROWS:
                action, count = _merge(grid, body, edit)
                touch(edit.table, cast(int, edit.row))
                touch(edit.table, cast(int, edit.row) + 1)
                actions.append(action)
                changed += count
            case EditOp.ROW_TO_HEADER:
                label = _label(grid, edit)
                if _norm(edit.old) != label:
                    raise EditError(f"r{edit.row} starts with {label!r}, not {edit.old!r}")
                touch(edit.table, cast(int, edit.row))
                to_header[edit.table].add(cast(int, edit.row))
                changed += sum(slot.origin for slot in grid[cast(int, edit.row)].values())
            case EditOp.HEADER_TO_ROW:
                heads = _header_rows(table.element)
                if edit.row is None or not 0 <= edit.row < len(heads):
                    raise EditError(f"no header row h{edit.row}")
                first = _cells(heads[edit.row])
                text = inner_html(first[0]) if first else ""
                if _norm(edit.old) != text:
                    raise EditError(f"h{edit.row} starts with {text!r}, not {edit.old!r}")
                to_body[edit.table].add(edit.row)
                changed += len(first)
    for (t, r), group in moves.items():
        if (t, r) in touched:
            raise EditError(f"body row {r} is changed by more than one row edit")
        actions.append(_moves(grids[t][r], rows[t][r], group, widths[t]))
    for t, moved in to_header.items():
        if moved != set(range(len(moved))):
            raise EditError("row_to_header moves only the top body rows (r0, or r0 and r1, ...)")
        top = [rows[t][i] for i in sorted(moved)]
        actions.append(lambda table=tables[t].element, top=top: _rows_to_header(table, top))
    for t, moved in to_body.items():
        heads = _header_rows(tables[t].element)
        if moved != set(range(len(heads) - len(moved), len(heads))):
            raise EditError("header_to_row moves only the bottom header rows")
        actions.append(_header_to_rows(tables[t].element, heads, sorted(moved)))
    for action in actions:
        action()
    return join_draft(tables), changed


def _label(grid: list[dict[int, _Slot]], edit: Edit) -> str:
    if edit.row is None or not 0 <= edit.row < len(grid):
        raise EditError(f"no body row {edit.row}")
    first = grid[edit.row].get(0)
    return inner_html(first.cell) if first is not None else ""


def _single(slot: _Slot, row: HtmlElement) -> bool:
    """A slot whose cell starts in `row` and covers one grid position."""
    return (
        slot.origin and slot.cell.getparent() is row and _span(slot.cell, "rowspan") == _span(slot.cell, "colspan") == 1
    )


def _width(grid: list[dict[int, _Slot]]) -> int:
    return max((max(line) + 1 for line in grid if line), default=0)


def _pad(line: dict[int, _Slot], row: HtmlElement, width: int) -> None:
    """Append empty cells to a row that ends short of `width` (a dropped cell leaves `no cell covers` issues)."""
    for col in range(max(line, default=-1) + 1, width):
        cell = lxml.html.fragment_fromstring("<td></td>")
        row.append(cell)
        line[col] = _Slot(cell, True)


def _moves(line: dict[int, _Slot], row: HtmlElement, group: list[Edit], width: int) -> Callable[[], None]:
    """Move contents within one body row, all at once: a target is empty or is itself moved away. A row that ends
    short of the table width is first padded with empty cells, so content can move into its missing positions."""
    sources = [cast(int, e.col) for e in group]
    targets = [e.to_col for e in group]
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise EditError("two moves in one row share a source or a target")
    _pad(line, row, width)  # appends only, so no other address changes
    for edit in group:
        for col in (edit.col, edit.to_col):
            if col is None or col not in line or not _single(line[col], row):
                raise EditError(f"r{edit.row} c{col} is not a single cell of that row")
        source = line[cast(int, edit.col)].cell
        if _norm(edit.old) != inner_html(source):
            raise EditError(f"r{edit.row} c{edit.col} holds {inner_html(source)!r}, not {edit.old!r}")
        if edit.to_col == edit.col:
            raise EditError(f"r{edit.row} c{edit.col} moves onto itself")
        target = line[cast(int, edit.to_col)].cell
        if inner_html(target) and edit.to_col not in sources:
            raise EditError(f"r{edit.row} c{edit.to_col} is not empty: {inner_html(target)!r}")
    contents = {cast(int, e.col): inner_html(line[cast(int, e.col)].cell) for e in group}

    def action() -> None:
        for col in sources:
            _set_content(line[col].cell, "")
        for edit in group:
            _set_content(line[cast(int, edit.to_col)].cell, contents[cast(int, edit.col)])

    return action


def _shift(line: dict[int, _Slot], row: HtmlElement, edit: Edit, width: int) -> Callable[[], None]:
    """Move the content from column `col` to the row's end so that `col` lands on `to_col`, keeping every value.

    Right: empty cells are inserted at `col`; cells pushed past the table width must be empty and are dropped.
    Left: the cells between `to_col` and `col` must be empty and are removed; the row is padded back to its width.
    Every position from the shift's start to the row's end must be a single cell of this row.
    """
    col, to_col = edit.col, edit.to_col
    end = max(line, default=-1) + 1
    if col is None or to_col is None or col == to_col or not 0 <= col <= end or to_col < 0:
        raise EditError(f"cannot shift r{edit.row} from c{col} to c{to_col}")
    start = min(col, to_col)
    if not all(_single(line[c], row) for c in range(start, end)):
        raise EditError(f"r{edit.row} has spanning or covered cells from c{start} on")
    current = inner_html(line[col].cell) if col < end else ""
    if _norm(edit.old) != current:
        raise EditError(f"r{edit.row} c{col} holds {current!r}, not {edit.old!r}")
    anchor = line[col].cell if col < end else None
    if to_col > col:
        k = to_col - col
        excess = max(0, end + k - max(width, end))
        dropped = [line[c].cell for c in range(end - excess, end)] if end - excess >= col else []
        if end - excess < col or any(inner_html(cell) for cell in dropped):
            raise EditError(f"shifting r{edit.row} by {k} pushes values past the table width {width}")

        def right() -> None:
            for _ in range(k):
                cell = lxml.html.fragment_fromstring("<td></td>")
                if anchor is not None:
                    anchor.addprevious(cell)
                else:
                    row.append(cell)
            for cell in dropped:
                _remove(cell)

        return right
    gap = [line[c].cell for c in range(to_col, col)]
    if any(inner_html(cell) for cell in gap):
        raise EditError(f"r{edit.row} c{to_col}..c{col - 1} are not empty; nothing can shift into them")
    refill = max(0, min(len(gap), width - (end - len(gap))))

    def left() -> None:
        for cell in gap:
            _remove(cell)
        for _ in range(refill):
            row.append(lxml.html.fragment_fromstring("<td></td>"))

    return left


_JOIN = re.compile(r"[\s\-\u00ad]+")


def _merge(grid: list[dict[int, _Slot]], body: list[HtmlElement], edit: Edit) -> tuple[Callable[[], None], int]:
    """Merge body row `row + 1` into `row`: a label wrapped over two rows, only one of which has values."""
    if edit.row is None or not 0 <= edit.row < len(grid) - 1:
        raise EditError(f"no body rows r{edit.row} and r{(edit.row or 0) + 1} to merge")
    upper, lower = body[edit.row], body[edit.row + 1]
    for line, row in ((grid[edit.row], upper), (grid[edit.row + 1], lower)):
        if not all(_single(slot, row) for slot in line.values()) or 0 not in line:
            raise EditError(f"r{edit.row} or r{edit.row + 1} has spanning cells; merge_rows needs plain rows")
    first, second = grid[edit.row][0].cell, grid[edit.row + 1][0].cell
    if _norm(edit.old) != inner_html(first):
        raise EditError(f"r{edit.row} starts with {inner_html(first)!r}, not {edit.old!r}")

    def has_values(line: dict[int, _Slot]) -> bool:
        return any(inner_html(slot.cell) for col, slot in line.items() if col > 0)

    if has_values(grid[edit.row]) and has_values(grid[edit.row + 1]):
        raise EditError(f"r{edit.row} and r{edit.row + 1} both have values; they are two rows")
    joined = _norm(f"{inner_html(first)} {inner_html(second)}")
    label = joined if edit.new is None else _norm(edit.new)
    if _JOIN.sub("", label) != _JOIN.sub("", joined):
        raise EditError(f"the merged label {label!r} is not the two labels joined")
    keep, drop = (lower, upper) if has_values(grid[edit.row + 1]) else (upper, lower)
    kept_label = _cells(keep)[0]
    level = first.get("data-level")

    def action() -> None:
        _set_content(kept_label, label)
        if level is not None:
            kept_label.set("data-level", level)
        _remove(drop)

    return action, 1


def _rows_to_header(table: HtmlElement, moved: list[HtmlElement]) -> None:
    heads = cast(list[HtmlElement], table.xpath("./thead"))
    if heads:
        head = heads[0]
    else:
        head = lxml.html.fragment_fromstring("<thead></thead>")
        table.insert(0, head)
    for row in moved:
        _remove(row)
        for cell in _cells(row):
            cell.tag = "th"
            cell.attrib.pop("data-level", None)
        head.append(row)


def _header_to_rows(table: HtmlElement, heads: list[HtmlElement], moved: list[int]) -> Callable[[], None]:
    grid = _grid(heads)
    width = max((max(line) + 1 for line in grid if line), default=0)
    for index in moved:
        if not all(slot.cell.getparent() is heads[index] for slot in grid[index].values()):
            raise EditError(f"h{index} is covered by a header cell from a row above")
        if any(_span(cell, "rowspan") > 1 for cell in _cells(heads[index])):
            raise EditError(f"h{index} has a cell spanning rows")

    def action() -> None:
        rows = [heads[i] for i in moved]
        bodies = cast(list[HtmlElement], table.xpath("./tbody"))
        if bodies:
            body = bodies[0]
        else:
            body = lxml.html.fragment_fromstring("<tbody></tbody>")
            table.append(body)
        for position, row in enumerate(rows):
            _remove(row)
            cells = _cells(row)
            for cell in cells:
                cell.tag = "td"
            if cells:
                cells[0].set("data-level", "0")
                if len(cells) == 1 and _span(cells[0], "colspan") == width:  # a heading across the table
                    cells[0].attrib.pop("colspan", None)
                    for _ in range(width - 1):
                        row.append(lxml.html.fragment_fromstring("<td></td>"))
            body.insert(position, row)
        for head in cast(list[HtmlElement], table.xpath("./thead")):
            if not _cells_in(head):
                _remove(head)

    return action


def _cells_in(head: HtmlElement) -> bool:
    return any(_cells(row) for row in cast(list[HtmlElement], head.xpath("./tr")))


def _slot(grid: list[dict[int, _Slot]], edit: Edit) -> _Slot:
    if edit.row is None or edit.col is None or not 0 <= edit.row < len(grid) or edit.col not in grid[edit.row]:
        raise EditError(f"no body position r{edit.row} c{edit.col}")
    slot = grid[edit.row][edit.col]
    if not slot.origin:
        raise EditError(f"r{edit.row} c{edit.col} is covered by a cell that starts elsewhere")
    return slot


def _replace_header(table: HtmlElement, header: HtmlElement | None) -> None:
    for old in cast(list[HtmlElement], table.xpath("./thead")):
        table.remove(old)
    if header is not None:
        table.insert(0, header)


def _insert_row(table: HtmlElement, anchor: HtmlElement | None, parent: HtmlElement | None, row: HtmlElement) -> None:
    if anchor is not None:
        anchor.addprevious(row)
    elif parent is not None:
        parent.append(row)
    else:
        bodies = cast(list[HtmlElement], table.xpath("./tbody"))
        if bodies:
            bodies[0].append(row)
        else:
            body = lxml.html.fragment_fromstring("<tbody></tbody>")
            body.append(row)
            table.append(body)


def _remove(element: HtmlElement) -> None:
    parent = element.getparent()
    if parent is not None:
        parent.remove(element)


def _interpret(region_id: str, text: str) -> RegionResult:
    return interpret(region_id, OcrResult(text=text, model="review"))


def _blocking(result: RegionResult) -> int:
    return sum(i.blocking for i in result.issues) + sum(i.blocking for t in result.tables for i in t.issues)


def _all_ready(result: RegionResult) -> bool:
    return result.status is RegionStatus.TABLES and all(t.outcome.value == "ready" for t in result.tables)


def judge(region_id: str, draft: str, answer: ReviewAnswer, *, max_changed_share: float = 0.3) -> ReviewOutcome:
    """The transcription to keep after `answer`, under the checks of the module docstring."""
    if answer.verdict is Verdict.UNRESOLVED:
        return ReviewOutcome(OutcomeState.UNRESOLVED, draft)
    if answer.verdict is Verdict.UNCHANGED or not answer.edits:
        reason = "edits given with verdict unchanged; ignored" if answer.edits else ""
        return ReviewOutcome(OutcomeState.UNCHANGED, draft, reason)
    try:
        revised, changed = apply_edits(draft, answer.edits)
    except EditError as error:
        return ReviewOutcome(OutcomeState.REJECTED, draft, f"edit rejected: {error}")
    tables = split_draft(draft) or []
    cells = sum(sum(s.origin for line in body_grid(t.element) for s in line.values()) for t in tables)
    if changed > max(1, max_changed_share * cells):
        return ReviewOutcome(OutcomeState.REJECTED, draft, f"changes {changed} of {cells} body cells", changed)
    before, after = _interpret(region_id, draft), _interpret(region_id, revised)
    if after.status is not RegionStatus.TABLES or len(after.tables) != len(before.tables):
        return ReviewOutcome(OutcomeState.REJECTED, draft, "the revision changes the number of tables", changed)
    if _blocking(after) > _blocking(before):
        reason = f"blocking issues rise from {_blocking(before)} to {_blocking(after)}"
        return ReviewOutcome(OutcomeState.REJECTED, draft, reason, changed)
    if _blocking(before) and _blocking(after) == _blocking(before):
        return ReviewOutcome(OutcomeState.REJECTED, draft, f"blocking issues stay at {_blocking(before)}", changed)
    if _all_ready(before) and not _all_ready(after):
        return ReviewOutcome(OutcomeState.REJECTED, draft, "a ready table is no longer ready", changed)
    return ReviewOutcome(OutcomeState.REVISED, revised, changed_cells=changed)


def agree(answers: Sequence[ReviewAnswer]) -> ReviewAnswer:
    """The first answer, keeping a `set_cell` edit only when every other answer proposes the same new content at
    the same position. Structural edits (header, title, levels, rows) come from the first answer alone: value
    edits are where a single review invents digits, header replacements are where it helps most."""
    first = answers[0]
    if first.verdict is not Verdict.REVISED:
        return first

    def key(edit: Edit) -> tuple[int, int | None, int | None, str]:
        return (edit.table, edit.row, edit.col, _norm(edit.new))

    others = [{key(e) for e in a.edits if e.op is EditOp.SET_CELL} for a in answers[1:]]
    kept = tuple(e for e in first.edits if e.op is not EditOp.SET_CELL or all(key(e) in o for o in others))
    return replace(first, edits=kept, verdict=Verdict.REVISED if kept else Verdict.UNCHANGED)


# --- a reviewer through OpenRouter ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE = {"type": "ephemeral"}


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _flat_usage(usage: dict[str, Any], prefix: str = "") -> dict[str, int]:
    """OpenRouter's usage object with nested details flattened (`completion_tokens_details.reasoning_tokens`)."""
    flat: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, dict):
            flat.update(_flat_usage(cast(dict[str, Any], value), f"{prefix}{key}."))
        elif isinstance(value, int) and not isinstance(value, bool):
            flat[f"{prefix}{key}"] = value
    return flat


def review_inputs(request: ReviewRequest, max_edge: int = 2576) -> tuple[str, Image.Image]:
    """What the reviewer is shown besides the prompt: the draft's view (with its parser issues) and the crop, resized
    as for the OCR model. Fine-tuning builds its examples from the same function."""
    return render_view(request.draft, draft_issues(request.region_id, request.draft)), model_image(
        request.image, max_edge
    )


class OpenRouterReviewer:
    """Reviews drafts with a model on OpenRouter's chat completions API, answering in `ANSWER_SCHEMA`.

    The key is read from `OPENROUTER_API_KEY` unless given. Requests ask for strict structured output and are
    routed only to providers that support every parameter sent, and, with `quantizations`, only to providers serving
    the model at those precisions (open-weight models are served from fp4 to bf16); `provider_sort="price"` tries
    the cheapest of them first. For Anthropic models the system prompt (`prompts/review.md`) is marked for caching.
    The answering provider is kept in the answer's settings, and the reasoning text in the answer when the provider
    returns it.

    With `url` pointing at another OpenAI-compatible server (a local `vllm serve`), the OpenRouter-only fields
    (provider routing, reasoning effort, usage accounting) are left out, no key is needed, and
    `chat_template_kwargs` (e.g. `{"enable_thinking": True}`) are passed to the server's chat template.
    Operational failures (transport, rate limits, server errors, a cut-off or filtered answer, an
    answer outside the schema) raise `ComponentFailure`.
    """

    def __init__(
        self,
        *,
        model: str = "anthropic/claude-haiku-5.5",
        effort: str = "medium",
        max_tokens: int = 32000,
        max_edge: int = 2576,
        quantizations: Sequence[str] = (),
        provider_sort: str | None = None,
        url: str = OPENROUTER_URL,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.quantizations = tuple(quantizations)
        self.provider_sort = provider_sort
        self.max_tokens = max_tokens
        self.max_edge = max_edge
        self.url = url
        self.local = url != OPENROUTER_URL
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        key = key or ("local" if self.local else "")
        if not key:
            raise ValueError("set OPENROUTER_API_KEY")
        self._client = httpx.AsyncClient(
            timeout=timeout, transport=transport,
            headers={"Authorization": f"Bearer {key}", "X-Title": "tablereader review trial"},
        )  # fmt: skip

    def request(self, request: ReviewRequest) -> dict[str, Any]:
        view, picture = review_inputs(request, self.max_edge)
        image = _png_data_url(picture)
        system: dict[str, Any] = {"type": "text", "text": PROMPT}
        if self.model.startswith("anthropic/"):
            system["cache_control"] = CACHE
        provider: dict[str, Any] = {"require_parameters": True}
        if self.quantizations:
            provider["quantizations"] = list(self.quantizations)
        if self.provider_sort:
            provider["sort"] = self.provider_sort
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": [system]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                             {"type": "text", "text": view}]},
            ],
            "reasoning": {"effort": self.effort},
            "response_format": {
                "type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": ANSWER_SCHEMA},
            },
            "provider": provider,
            "usage": {"include": True},
        }  # fmt: skip
        if self.local:
            for openrouter_only in ("reasoning", "provider", "usage"):
                del body[openrouter_only]
            if self.chat_template_kwargs:
                body["chat_template_kwargs"] = self.chat_template_kwargs
        return body

    async def review(self, request: ReviewRequest) -> ReviewAnswer:
        try:
            response = await self._client.post(self.url, json=self.request(request))
        except httpx.TransportError as error:
            raise ComponentFailure("transport", repr(error), retryable=True) from error
        if response.status_code == 429 or response.status_code >= 500:
            raise ComponentFailure("server", f"{response.status_code} {response.text[:300]}", retryable=True)
        if response.status_code >= 400:
            raise ComponentFailure("bad_request", f"{response.status_code} {response.text[:300]}", retryable=False)
        try:
            body: dict[str, Any] = response.json()
            if "error" in body:  # errors after routing can arrive with status 200
                raise ComponentFailure("provider", json.dumps(body["error"])[:300], retryable=True)
            choice: dict[str, Any] = body["choices"][0]
            finish = choice.get("finish_reason")
            text: str = choice["message"]["content"] or ""
            message: dict[str, Any] = choice["message"]
            reasoning: str = message.get("reasoning") or message.get("reasoning_content") or ""  # vLLM: either
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ComponentFailure("malformed_answer", str(error), retryable=True) from error
        if finish != "stop":
            raise ComponentFailure(f"finish_{finish}", f"answer stopped: {finish}", retryable=False)
        usage_body: dict[str, Any] = body.get("usage") or {}
        settings = {
            "effort": None if self.local else self.effort,
            "max_tokens": self.max_tokens,
            "max_edge": self.max_edge,
            "via": self.url if self.local else "openrouter",
            "prompt_sha256": PROMPT_SHA256,
            "provider": body.get("provider"),
            "quantizations": list(self.quantizations),
            "provider_sort": self.provider_sort,
            **({"chat_template_kwargs": self.chat_template_kwargs} if self.local else {}),
        }
        try:
            data = json.loads(text)
            return ReviewAnswer(
                verdict=Verdict(data["verdict"]), edits=tuple(Edit.from_json(e) for e in data["edits"]),
                model=str(body.get("model", self.model)), settings=settings, usage=_flat_usage(usage_body),
                cost=float(usage_body["cost"]) if isinstance(usage_body.get("cost"), int | float) else None, raw=text,
                reasoning=reasoning,
            )  # fmt: skip
        except (ValueError, KeyError, TypeError) as error:
            raise ComponentFailure("malformed_answer", str(error), retryable=True) from error

    async def aclose(self) -> None:
        await self._client.aclose()
