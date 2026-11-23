"""LLM review: the draft view, guarded edits and the acceptance checks, without calling a model."""

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from evaluation.records import crop_records
from evaluation.review import (
    ANSWER_SCHEMA,
    PROMPT,
    STRUCTURE_OPS,
    Edit,
    EditError,
    EditOp,
    OpenRouterReviewer,
    OutcomeState,
    ReviewAnswer,
    ReviewRequest,
    Verdict,
    _interpret,  # pyright: ignore[reportPrivateUsage]
    agree,
    apply_edits,
    body_grid,
    inner_html,
    judge,
    render_view,
    split_draft,
)
from extraction.results import ComponentFailure, RegionResult

ROOT = Path(__file__).resolve().parents[3]

DRAFT = """# Table 5. Receipts
<table>
<thead><tr><th>Item</th><th colspan="2">Amount</th></tr><tr><th></th><th>2024</th><th>2025</th></tr></thead>
<tbody>
<tr><td data-level="0">Receipts:</td><td></td><td></td></tr>
<tr><td data-level="1">Taxes</td><td>1,204</td><td>1,310<sup>1</sup></td></tr>
<tr><td data-level="1" rowspan="2">Fees</td><td>88</td><td>91</td></tr>
<tr><td>7</td><td>-</td></tr>
</tbody>
<tfoot><tr><td colspan="3">1/ Provisional.</td></tr></tfoot>
</table>"""

BROKEN_HEADER = DRAFT.replace('<th colspan="2">Amount</th>', '<th colspan="3">Amount</th>')


def answer(*edits: Edit, verdict: Verdict = Verdict.REVISED) -> ReviewAnswer:
    return ReviewAnswer(verdict=verdict, edits=edits, model="test")


def records(text: str) -> frozenset[tuple[tuple[str, ...], str, str]]:
    result: RegionResult = _interpret("r", text)
    return crop_records([t.candidate for t in result.tables if t.candidate is not None])


def test_view_numbers_body_positions_and_marks_covered_ones() -> None:
    view = render_view(DRAFT, ["note: something"])
    assert 'Title: "Table 5. Receipts"' in view
    assert 'r1 level=1 | c0 "Taxes" | c1 "1,204" | c2 "1,310<sup>1</sup>"' in view
    assert 'r3 level=none | c0 ^ | c1 "7" | c2 "-"' in view
    assert '"1/ Provisional."' in view and "- note: something" in view


def test_draft_round_trips_without_edits() -> None:
    tables = split_draft(DRAFT)
    assert tables is not None
    revised, changed = apply_edits(DRAFT, [])
    assert changed == 0 and records(revised) == records(DRAFT)


@pytest.mark.parametrize(
    "text", ["NOT_A_TABLE", "<table><tr><td>1</td></tr>", "Some prose\n<table><tr><td>1</td></tr></table>"]
)
def test_drafts_that_would_lose_text_are_not_reviewable(text: str) -> None:
    assert split_draft(text) is None


def test_set_cell_changes_only_the_named_cell() -> None:
    revised, changed = apply_edits(DRAFT, [Edit(EditOp.SET_CELL, 0, row=1, col=1, old="1,204", new="1,284")])
    assert changed == 1
    gone, added = records(DRAFT) - records(revised), records(revised) - records(DRAFT)
    assert {r[2] for r in gone} == {"1,204"} and {r[2] for r in added} == {"1,284"}


def test_set_cell_keeps_markers_written_as_sup() -> None:
    revised, _ = apply_edits(DRAFT, [Edit(EditOp.SET_CELL, 0, row=2, col=2, old="91", new="97<sup>2</sup>")])
    assert "<td>97<sup>2</sup></td>" in revised


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (Edit(EditOp.SET_CELL, 0, row=1, col=1, old="1,205", new="1"), "holds '1,204'"),
        (Edit(EditOp.SET_CELL, 0, row=3, col=0, old="Fees", new="x"), "covered"),
        (Edit(EditOp.SET_CELL, 0, row=9, col=0, old="", new="x"), "no body position"),
        (Edit(EditOp.SET_CELL, 1, row=0, col=0, old="", new="x"), "no table 1"),
        (Edit(EditOp.SET_LEVEL, 0, row=1, old="0", new="2"), "has level '1'"),
        (Edit(EditOp.SET_TITLE, 0, old="Table 6", new="x"), "has title"),
        (Edit(EditOp.DELETE_ROW, 0, row=1, old="Fees"), "starts with 'Taxes'"),
        (Edit(EditOp.REPLACE_HEADER, 0, new="<th>x</th>"), "not exactly one <thead>"),
        (Edit(EditOp.INSERT_ROW, 0, row=1, new="<thead><tr><th>x</th></tr></thead>"), "not exactly one <tr>"),
    ],
)
def test_guards_reject_edits_that_do_not_match_the_draft(edit: Edit, message: str) -> None:
    with pytest.raises(EditError, match=message):
        apply_edits(DRAFT, [edit])


def test_a_shift_beyond_the_row_is_refused() -> None:
    with pytest.raises(EditError, match="pushes values past the table width"):
        apply_edits(DRAFT, [Edit(EditOp.SHIFT_CELLS, 0, row=1, col=1, to_col=9, old="1,204")])


def test_addresses_refer_to_the_draft_even_after_a_row_is_inserted() -> None:
    edits = [
        Edit(EditOp.INSERT_ROW, 0, row=0, new='<tr><td data-level="0">Total</td><td>1,292</td><td>1,401</td></tr>'),
        Edit(EditOp.SET_CELL, 0, row=1, col=1, old="1,204", new="1,205"),
    ]
    revised, _ = apply_edits(DRAFT, edits)
    assert revised.index("Total") < revised.index("Receipts:") and "1,205" in revised


def test_replace_header_fixes_spans_and_judge_accepts_it() -> None:
    header = (
        '<thead><tr><th>Item</th><th colspan="2">Amount</th></tr><tr><th></th><th>2024</th><th>2025</th></tr></thead>'
    )
    outcome = judge("r", BROKEN_HEADER, answer(Edit(EditOp.REPLACE_HEADER, 0, new=header)))
    assert outcome.state is OutcomeState.REVISED
    assert records(outcome.text) == records(DRAFT)


def test_replace_header_accepts_bare_header_rows() -> None:
    rows = '<tr><th>Item</th><th colspan="2">Amount</th></tr><tr><th></th><th>2024</th><th>2025</th></tr>'
    outcome = judge("r", BROKEN_HEADER, answer(Edit(EditOp.REPLACE_HEADER, 0, new=rows)))
    assert outcome.state is OutcomeState.REVISED
    assert records(outcome.text) == records(DRAFT)


@pytest.mark.parametrize("new", ["<th>Item</th><th>2024</th>", "Target"])
def test_replace_header_refuses_other_fragments(new: str) -> None:
    with pytest.raises(EditError, match="exactly one <thead>"):
        apply_edits(BROKEN_HEADER, [Edit(EditOp.REPLACE_HEADER, 0, new=new)])


@pytest.mark.parametrize("new", [None, "<tr><th>Item</th><th>2024</th><th>2025</th></tr>"])
def test_replace_header_with_cell_fields_is_refused(new: str | None) -> None:
    with pytest.raises(EditError, match="no row, column or old value"):
        apply_edits(DRAFT, [Edit(EditOp.REPLACE_HEADER, 0, row=0, col=1, old="", new=new)])
    revised, _ = apply_edits(DRAFT, [Edit(EditOp.REPLACE_HEADER, 0)])
    assert "<thead" not in revised


def test_one_header_replacement_per_table() -> None:
    rows = ["<tr><th>Item</th><th>2024</th><th>2025</th></tr>", "<tr><th></th><th>$</th><th>$</th></tr>"]
    with pytest.raises(EditError, match="more than one replace_header"):
        apply_edits(DRAFT, [Edit(EditOp.REPLACE_HEADER, 0, new=row) for row in rows])


def test_judge_rejects_a_revision_with_more_blocking_issues() -> None:
    bad = '<thead><tr><th>Item</th><th colspan="5">Amount</th></tr></thead>'
    outcome = judge("r", DRAFT, answer(Edit(EditOp.REPLACE_HEADER, 0, new=bad)))
    assert outcome.state is OutcomeState.REJECTED and outcome.text == DRAFT


def test_judge_rejects_a_revision_that_leaves_blocking_issues_in_place() -> None:
    title = Edit(EditOp.SET_TITLE, 0, old="Table 5. Receipts", new="Table 5. Receipts and fees")
    outcome = judge("r", BROKEN_HEADER, answer(title))
    assert outcome.state is OutcomeState.REJECTED and outcome.reason.startswith("blocking issues stay at")
    assert judge("r", DRAFT, answer(title)).state is OutcomeState.REVISED


def test_judge_rejects_rewrites() -> None:
    edits = [
        Edit(EditOp.SET_CELL, 0, row=1, col=c, old=v, new="0") for c, v in ((1, "1,204"), (2, "1,310<sup>1</sup>"))
    ]
    edits += [Edit(EditOp.SET_CELL, 0, row=2, col=c, old=v, new="0") for c, v in ((1, "88"), (2, "91"))]
    outcome = judge("r", DRAFT, answer(*edits))
    assert outcome.state is OutcomeState.REJECTED and "changes 4 of" in outcome.reason


def test_judge_keeps_the_draft_when_one_guard_fails() -> None:
    edits = (Edit(EditOp.SET_CELL, 0, row=1, col=1, old="1,204", new="1,205"), Edit(EditOp.SET_TITLE, 0, old="x"))
    outcome = judge("r", DRAFT, answer(*edits))
    assert outcome.state is OutcomeState.REJECTED and outcome.text == DRAFT


@pytest.mark.parametrize(("verdict", "state"), [(Verdict.UNCHANGED, "unchanged"), (Verdict.UNRESOLVED, "unresolved")])
def test_verdicts_without_edits_keep_the_draft(verdict: Verdict, state: str) -> None:
    outcome = judge("r", DRAFT, answer(verdict=verdict))
    assert outcome.state.value == state and outcome.text == DRAFT


def test_prompt_carries_the_current_transcription_spec() -> None:
    doc = (ROOT / "docs/transcription-format.md").read_text(encoding="utf-8")
    spec = doc[doc.index("## Response shape") : doc.index("## Parser behavior")].rstrip() + "\n"
    assert PROMPT.endswith(spec), "rebuild prompts/review.md from docs/transcription-format.md"
    assert all(f"`{op.value}`" in PROMPT for op in STRUCTURE_OPS)
    assert "`set_cell`" not in PROMPT


def reviewer_with(handler: Any) -> OpenRouterReviewer:
    return OpenRouterReviewer(api_key="test", transport=httpx.MockTransport(handler))


def completion(content: str, finish: str = "stop", reasoning: str | None = None) -> dict[str, Any]:
    usage = {"prompt_tokens": 5000, "completion_tokens": 900, "cost": 0.00095,
             "completion_tokens_details": {"reasoning_tokens": 700}}  # fmt: skip
    message = {"content": content, "reasoning": reasoning}
    return {"model": "anthropic/claude-haiku-5.5", "usage": usage,
            "choices": [{"finish_reason": finish, "message": message}]}  # fmt: skip


def test_request_sends_the_image_the_view_and_the_schema() -> None:
    image = Image.new("RGB", (4000, 1000), "white")
    body = reviewer_with(lambda r: httpx.Response(500)).request(ReviewRequest("r", image, DRAFT))
    system, user = body["messages"]
    assert system["content"][0]["cache_control"] == {"type": "ephemeral"} and system["content"][0]["text"] == PROMPT
    url: str = user["content"][0]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.removeprefix("data:image/png;base64,"))))
    assert max(sent.size) == 2576 and user["content"][1]["text"].startswith("## Table 0")
    assert body["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA
    assert body["response_format"]["json_schema"]["strict"] and body["provider"] == {"require_parameters": True}
    json.dumps(body)  # the request is plain JSON


def test_answer_is_parsed_with_usage_and_cost() -> None:
    edit = {"op": "set_cell", "table": 0, "row": 1, "col": 1, "old": "1,204", "new": "1,284", "reason": "image"}
    content = json.dumps({"verdict": "revised", "edits": [edit]})
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion(content))

    answer = asyncio.run(reviewer_with(handler).review(ReviewRequest("r", Image.new("RGB", (10, 10)), DRAFT)))
    assert seen[0].headers["Authorization"] == "Bearer test"
    assert answer.verdict is Verdict.REVISED and answer.edits[0] == Edit(
        EditOp.SET_CELL, 0, 1, 1, "1,204", "1,284", "image"
    )
    assert answer.usage["completion_tokens_details.reasoning_tokens"] == 700 and answer.cost == pytest.approx(0.00095)
    assert judge("r", DRAFT, answer).state is OutcomeState.REVISED
    assert answer.reasoning == ""


def test_reasoning_text_is_kept() -> None:
    content = json.dumps({"verdict": "unchanged", "edits": []})
    reply = completion(content, reasoning="The header spans line up with the columns.")
    reviewer = reviewer_with(lambda r: httpx.Response(200, json=reply))
    answer = asyncio.run(reviewer.review(ReviewRequest("r", Image.new("RGB", (10, 10)), DRAFT)))
    assert answer.reasoning == "The header spans line up with the columns."


@pytest.mark.parametrize(
    ("response", "code", "retryable"),
    [
        (httpx.Response(429, text="slow down"), "server", True),
        (httpx.Response(400, text="bad"), "bad_request", False),
        (httpx.Response(200, json={"error": {"message": "provider down"}}), "provider", True),
        (httpx.Response(200, json=completion("{}", finish="length")), "finish_length", False),
        (httpx.Response(200, json=completion("not json")), "malformed_answer", True),
    ],
)
def test_operational_failures_raise(response: httpx.Response, code: str, retryable: bool) -> None:
    reviewer = reviewer_with(lambda r: response)
    with pytest.raises(ComponentFailure) as caught:
        asyncio.run(reviewer.review(ReviewRequest("r", Image.new("RGB", (10, 10)), DRAFT)))
    assert caught.value.code == code and caught.value.retryable is retryable


def test_a_missing_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterReviewer()


def test_agreement_keeps_value_edits_only_when_every_review_proposes_them() -> None:
    header = Edit(EditOp.REPLACE_HEADER, 0, new="<thead><tr><th>Item</th><th>2024</th><th>2025</th></tr></thead>")
    both = Edit(EditOp.SET_CELL, 0, row=1, col=1, old="1,204", new="1,284")
    only_first = Edit(EditOp.SET_CELL, 0, row=2, col=1, old="88", new="83")
    other = Edit(EditOp.SET_CELL, 0, row=2, col=1, old="88", new="86")
    combined = agree([answer(header, both, only_first), answer(Edit(both.op, 0, 1, 1, "1,204", " 1,284 "), other)])
    assert combined.edits == (header, both) and combined.verdict is Verdict.REVISED
    alone = agree([answer(only_first), answer(verdict=Verdict.UNRESOLVED)])
    assert alone.edits == () and alone.verdict is Verdict.UNCHANGED
    assert agree([answer(verdict=Verdict.UNRESOLVED), answer(both)]).verdict is Verdict.UNRESOLVED


STRUCT = """<table>
<thead><tr><th>Item</th><th>2023</th><th>2024</th><th>2025</th></tr><tr><th colspan="4">I.—Reports</th></tr></thead>
<tbody>
<tr><td data-level="0"></td><td>1,000</td><td>1,000</td><td>1,000</td></tr>
<tr><td data-level="0">Married, widowed, or</td><td></td><td></td><td></td></tr>
<tr><td data-level="1">divorced</td><td>19,720</td><td>21,045</td><td>22,310</td></tr>
<tr><td data-level="0">Single</td><td>12,550</td><td>8,933</td><td></td></tr>
</tbody>
</table>"""


def body(text: str) -> list[list[str]]:
    tables = split_draft(text)
    assert tables is not None
    return [[inner_html(slot.cell) for _, slot in sorted(line.items())] for line in body_grid(tables[0].element)]


def test_view_numbers_header_rows() -> None:
    view = render_view(STRUCT)
    assert 'h1 <tr><th colspan="4">I.—Reports</th></tr>' in view and "Header rows:" in view


def test_moves_shift_a_run_of_values_without_retyping_them() -> None:
    edits = [
        Edit(EditOp.MOVE_CELL, 0, row=3, col=2, to_col=3, old="8,933"),
        Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=2, old="12,550"),
    ]
    revised, changed = apply_edits(STRUCT, edits)
    assert body(revised)[3] == ["Single", "", "12,550", "8,933"] and changed == 2


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=2, old="12,550"), "not empty"),
        (Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=3, old="12,551"), "holds '12,550'"),
        (Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=9, old="12,550"), "not a single cell"),
        (Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=1, old="12,550"), "onto itself"),
    ],
)
def test_move_guards(edit: Edit, message: str) -> None:
    with pytest.raises(EditError, match=message):
        apply_edits(STRUCT, [edit])


def test_moves_cannot_touch_spanning_cells() -> None:
    with pytest.raises(EditError, match="not a single cell"):
        apply_edits(DRAFT, [Edit(EditOp.MOVE_CELL, 0, row=2, col=0, to_col=1, old="Fees")])


def test_merge_joins_a_wrapped_label_and_keeps_the_values_and_first_level() -> None:
    revised, _ = apply_edits(STRUCT, [Edit(EditOp.MERGE_ROWS, 0, row=1, old="Married, widowed, or")])
    rows = body(revised)
    assert rows[1] == ["Married, widowed, or divorced", "19,720", "21,045", "22,310"] and len(rows) == 3
    assert '<td data-level="0">Married, widowed, or divorced</td>' in revised


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (Edit(EditOp.MERGE_ROWS, 0, row=2, old="divorced"), "both have values"),
        (
            Edit(EditOp.MERGE_ROWS, 0, row=1, old="Married, widowed, or", new="Married or divorced"),
            "not the two labels",
        ),
        (Edit(EditOp.MERGE_ROWS, 0, row=3, old="Single"), "no body rows"),
    ],
)
def test_merge_guards(edit: Edit, message: str) -> None:
    with pytest.raises(EditError, match=message):
        apply_edits(STRUCT, [edit])


def test_merge_may_drop_a_line_break_hyphen() -> None:
    draft = STRUCT.replace("Married, widowed, or", "Acquit-").replace(">divorced<", ">tals<")
    revised, _ = apply_edits(draft, [Edit(EditOp.MERGE_ROWS, 0, row=1, old="Acquit-", new="Acquittals")])
    assert body(revised)[1][0] == "Acquittals"


def test_a_units_row_moves_into_the_header() -> None:
    revised, _ = apply_edits(STRUCT, [Edit(EditOp.ROW_TO_HEADER, 0, row=0, old="")])
    assert "<tr><th></th><th>1,000</th><th>1,000</th><th>1,000</th></tr></thead>" in "".join(revised.split())
    assert body(revised)[0][0] == "Married, widowed, or"
    with pytest.raises(EditError, match="only the top body rows"):
        apply_edits(STRUCT, [Edit(EditOp.ROW_TO_HEADER, 0, row=1, old="Married, widowed, or")])


def test_a_section_heading_moves_from_the_header_into_the_body() -> None:
    revised, _ = apply_edits(STRUCT, [Edit(EditOp.HEADER_TO_ROW, 0, row=1, old="I.—Reports")])
    assert body(revised)[0] == ["I.—Reports", "", "", ""]
    assert '<td data-level="0">I.—Reports</td>' in revised
    with pytest.raises(EditError, match="only the bottom header rows"):
        apply_edits(STRUCT, [Edit(EditOp.HEADER_TO_ROW, 0, row=0, old="Item")])


def test_a_row_is_changed_by_one_row_edit_at_most() -> None:
    edits = [
        Edit(EditOp.MERGE_ROWS, 0, row=1, old="Married, widowed, or"),
        Edit(EditOp.DELETE_ROW, 0, row=2, old="divorced"),
    ]
    with pytest.raises(EditError, match="more than one row edit"):
        apply_edits(STRUCT, edits)


def test_structure_edits_keep_every_value() -> None:
    edits = (
        Edit(EditOp.ROW_TO_HEADER, 0, row=0, old=""),
        Edit(EditOp.MERGE_ROWS, 0, row=1, old="Married, widowed, or"),
        Edit(EditOp.MOVE_CELL, 0, row=3, col=2, to_col=3, old="8,933"),
        Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=2, old="12,550"),
    )
    outcome = judge("r", STRUCT, answer(*edits), max_changed_share=1.0)
    assert outcome.state is OutcomeState.REVISED
    values = sorted(v for *_, v in records(outcome.text))
    assert values == ["12,550", "19,720", "21,045", "22,310", "8,933"]


SHORT_ROW = STRUCT.replace("<td>12,550</td><td>8,933</td><td></td>", "<td>12,550</td><td>8,933</td>")


def test_a_short_row_shifts_right_into_its_missing_cell() -> None:
    revised, changed = apply_edits(SHORT_ROW, [Edit(EditOp.SHIFT_CELLS, 0, row=3, col=1, to_col=2, old="12,550")])
    assert body(revised)[3] == ["Single", "", "12,550", "8,933"] and changed == 1
    assert _interpret("r", revised).tables[0].outcome.value == "ready"


def test_a_full_row_shifts_right_only_into_trailing_empty_cells() -> None:
    revised, _ = apply_edits(STRUCT, [Edit(EditOp.SHIFT_CELLS, 0, row=3, col=1, to_col=2, old="12,550")])
    assert body(revised)[3] == ["Single", "", "12,550", "8,933"]
    with pytest.raises(EditError, match="pushes values past"):
        apply_edits(STRUCT, [Edit(EditOp.SHIFT_CELLS, 0, row=2, col=1, to_col=2, old="19,720")])


def test_a_row_shifts_left_over_empty_cells_and_keeps_its_width() -> None:
    draft = STRUCT.replace("<td>12,550</td><td>8,933</td><td></td>", "<td></td><td>12,550</td><td>8,933</td>")
    revised, _ = apply_edits(draft, [Edit(EditOp.SHIFT_CELLS, 0, row=3, col=2, to_col=1, old="12,550")])
    assert body(revised)[3] == ["Single", "12,550", "8,933", ""]
    with pytest.raises(EditError, match="are not empty"):
        apply_edits(STRUCT, [Edit(EditOp.SHIFT_CELLS, 0, row=3, col=2, to_col=1, old="8,933")])


def test_shift_guards() -> None:
    with pytest.raises(EditError, match="holds '12,550'"):
        apply_edits(STRUCT, [Edit(EditOp.SHIFT_CELLS, 0, row=3, col=1, to_col=2, old="12,551")])
    with pytest.raises(EditError, match="spanning or covered"):
        apply_edits(DRAFT, [Edit(EditOp.SHIFT_CELLS, 0, row=3, col=0, to_col=1, old="")])
    edits = [
        Edit(EditOp.SHIFT_CELLS, 0, row=3, col=1, to_col=2, old="12,550"),
        Edit(EditOp.MOVE_CELL, 0, row=3, col=2, to_col=3, old="8,933"),
    ]
    with pytest.raises(EditError, match="more than one row edit"):
        apply_edits(STRUCT, edits)


def test_moves_may_fill_the_missing_end_of_a_short_row() -> None:
    edits = [
        Edit(EditOp.MOVE_CELL, 0, row=3, col=2, to_col=3, old="8,933"),
        Edit(EditOp.MOVE_CELL, 0, row=3, col=1, to_col=2, old="12,550"),
    ]
    revised, _ = apply_edits(SHORT_ROW, edits)
    assert body(revised)[3] == ["Single", "", "12,550", "8,933"]


def test_open_weight_models_get_precision_routing_and_no_cache_marks() -> None:
    reviewer = OpenRouterReviewer(api_key="test", model="google/gemma-4-26b-a4b-it", quantizations=["bf16"])
    body = reviewer.request(ReviewRequest("r", Image.new("RGB", (10, 10)), DRAFT))
    assert body["provider"] == {"require_parameters": True, "quantizations": ["bf16"]}
    assert "cache_control" not in body["messages"][0]["content"][0]


def test_a_local_server_gets_no_openrouter_fields_and_its_reasoning_is_kept() -> None:
    seen: list[dict[str, Any]] = []
    content = json.dumps({"verdict": "unchanged", "edits": []})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        reply = completion(content)
        reply["choices"][0]["message"] = {"content": content, "reasoning_content": "Spans match."}
        return httpx.Response(200, json=reply)

    reviewer = OpenRouterReviewer(
        model="google/gemma-4-12B-it", url="http://localhost:8000/v1/chat/completions",
        chat_template_kwargs={"enable_thinking": True}, transport=httpx.MockTransport(handler),
    )  # fmt: skip
    answer = asyncio.run(reviewer.review(ReviewRequest("r", Image.new("RGB", (10, 10)), DRAFT)))
    body = seen[0]
    assert not {"provider", "reasoning", "usage"} & body.keys()
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "cache_control" not in body["messages"][0]["content"][0]
    assert answer.reasoning == "Spans match." and answer.settings["via"].startswith("http://localhost")
