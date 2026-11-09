"""What a read writes: the document result as JSON, one line per table, the crops, and a page to check by eye.

    <out>/document.json   source, pages, release, and per region its boxes, status, transcription and tables
    <out>/tables.jsonl    one table per line: ids, page, outcome, issues and the table JSON (format 0.3)
    <out>/crops/<region_id>.png
    <out>/review.html     each crop beside the tables read from it

The review page is built from the parsed transcription with every text escaped; model output is never inserted
as markup.
"""

import html
import json
from pathlib import Path
from typing import Any

from document_prep.models import Box
from extraction.results import encode_issue, encode_region_result
from table_core.codec import JsonObject, JsonValue, encode_table
from table_core.parsing import ParsedRow, ParsedText, parse_output
from tablereader.reader import DocumentResult, RegionRead


def _box(box: Box) -> list[JsonValue]:
    return [round(v, 2) for v in (box.left, box.top, box.right, box.bottom)]


def encode_region(read: RegionRead) -> JsonObject:
    prepared = read.prepared
    payload = encode_region_result(read.result)
    payload.update(
        page=prepared.region.page,
        box=_box(prepared.region.box),
        crop_box=_box(prepared.crop_box),
        crop=f"crops/{prepared.region.region_id}.png",
    )
    return payload


def encode_document(result: DocumentResult) -> JsonObject:
    return {
        "source": {"name": result.source.path.name, "sha256": result.source.sha256},
        "pages": list(result.pages),
        "release": result.release_id,
        "regions": [encode_region(r) for r in result.regions],
    }


def table_lines(result: DocumentResult) -> list[JsonObject]:
    lines: list[JsonObject] = []
    for read in result.regions:
        for table in read.result.tables:
            lines.append({
                "table_id": table.table_id, "region_id": table.region_id, "page": read.prepared.region.page,
                "outcome": table.outcome.value, "issues": [encode_issue(i) for i in table.issues],
                "table": encode_table(table.candidate) if table.candidate is not None else None,
            })  # fmt: skip
    return lines


def write_outputs(result: DocumentResult, out: Path) -> None:
    (out / "crops").mkdir(parents=True, exist_ok=True)
    for read in result.regions:
        read.prepared.image.save(out / "crops" / f"{read.prepared.region.region_id}.png")
    (out / "document.json").write_text(json.dumps(encode_document(result), ensure_ascii=False, indent=1))
    (out / "tables.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in table_lines(result)))
    (out / "review.html").write_text(review_page(result))


def _text(content: ParsedText) -> str:
    pieces: list[str] = []
    position = 0
    for marker in content.markers:
        pieces += [html.escape(content.text[position : marker.at]), f"<sup>{html.escape(marker.text)}</sup>"]
        position = marker.at
    pieces.append(html.escape(content.text[position:]))
    return "".join(pieces)


def _rows(rows: tuple[ParsedRow, ...]) -> str:
    out: list[str] = []
    for row in rows:
        cells: list[str] = []
        for cell in row.cells:
            tag = "th" if cell.header else "td"
            spans = (f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else "") + (
                f' colspan="{cell.colspan}"' if cell.colspan > 1 else ""
            )
            indent = f' style="padding-left:{0.3 + 1.2 * cell.level:.1f}em"' if cell.level else ""
            cells.append(f"<{tag}{spans}{indent}>{_text(cell.content)}</{tag}>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out)


def _region_html(read: RegionRead) -> str:
    result, region = read.result, read.prepared.region
    outcomes = ", ".join(t.outcome.value for t in result.tables) or result.status.value
    parts = [f'<section><h2>Page {region.page} · {html.escape(region.region_id)} · {html.escape(outcomes)}</h2>',
             f'<div class="pair"><img src="crops/{html.escape(region.region_id)}.png" alt=""><div>']  # fmt: skip
    if result.response is not None:
        for fragment in parse_output(result.response.text).fragments:
            table = fragment.table
            if table is None:
                parts.append("<p class=issue>unreadable table</p>")
                continue
            if table.title is not None:
                parts.append(f"<h3>{_text(table.title)}</h3>")
            parts.append(f"<table><thead>{_rows(table.head)}</thead><tbody>{_rows(table.body)}</tbody>"
                         f"<tfoot>{_rows(table.foot)}</tfoot></table>")  # fmt: skip
    if result.error:
        parts.append(f"<p class=issue>{html.escape(result.error)}</p>")
    issues = [i for t in result.tables for i in t.issues] + list(result.issues)
    if issues:
        items = "".join(f"<li>{html.escape(i.code)}: {html.escape(i.message)}</li>" for i in issues)
        parts.append(f"<ul class=issue>{items}</ul>")
    parts.append("</div></div></section>")
    return "".join(parts)


STYLE = """body{font:14px system-ui,sans-serif;margin:16px;color:#222;background:#fff}
.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}
img{max-width:100%;border:1px solid #ccc}table{border-collapse:collapse;font-size:12px}
td,th{border:1px solid #ccc;padding:2px 6px;vertical-align:top}th{background:#f3f3f3}
h2{font-size:15px;margin-top:32px}.issue{color:#a40}
@media (max-width:800px){.pair{grid-template-columns:1fr}}
@media (prefers-color-scheme:dark){body{background:#161616;color:#ddd}th{background:#262626}
td,th,img{border-color:#444}}"""


def review_page(result: DocumentResult) -> str:
    title = html.escape(result.source.path.name)
    body: Any = "".join(_region_html(r) for r in result.regions) or "<p>No tables found.</p>"
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f"<title>{title}</title><style>{STYLE}</style></head><body><h1>{title}</h1>"
            f"<p>{len(result.tables)} tables on {len(result.pages)} pages · model {html.escape(result.release_id)}</p>"
            f"{body}</body></html>")  # fmt: skip
