# OCR transcription format

Contract revision 1. This is the text the fine-tuned OCR model emits for one region image, and the only input format the parser accepts. The parser converts it into the [table JSON format](table-format.md). Neither format is selectable per request.

The model is trained to produce exactly this format, and a model release records the contract revision it was trained for. A release whose revision the code does not support is rejected when it is loaded. Changing this contract is a coordinated release: annotations, training targets, model and parser change together. There is no runtime format selector or fallback dialect.

## Response shape

A response is either exactly `NOT_A_TABLE`, or one or more HTML tables in reading order. Each table may be preceded by one Markdown heading of the form `# Title`. Nothing else is allowed: no explanatory prose, code fences, Markdown pipe tables, or `<caption>` elements.

```html
# Receipts (in dollars)
<table>
  <thead>
    <tr><th>Item</th><th>2025</th></tr>
  </thead>
  <tbody>
    <tr><td data-level="0">Receipts:</td><td></td></tr>
    <tr><td data-level="1">Taxes</td><td>100<sup>1</sup></td></tr>
  </tbody>
  <tfoot>
    <tr><td colspan="2">1/ Provisional figure.</td></tr>
  </tfoot>
</table>
```

## Constructs

| Construct | Meaning |
| --- | --- |
| `# ...` immediately before a table | That table's printed title, including a printed table number and units line when present, in source order. Absent when no title is printed. It is not inherited by later tables. |
| `thead`, `tr`, `th` | Rows printed as column headings; merged headers define the path to each leaf column. Absent when the table prints no column headings. |
| `tbody`, `tr`, `td` | Body rows and cells. The JSON conversion reads the first logical column as row labels. |
| `rowspan`, `colspan` | Positive integer spans; covered positions refer to one source cell |
| `data-level="n"` | Nonnegative indentation level of the first cell in a body row; `0` means flush left |
| `data-unclear="1"` | The model could not read this cell confidently; its text is a best reading |
| `sup` inside a title, header, or cell | A footnote-marker occurrence, distinct from the surrounding text |
| `tfoot` | Footnote lines, each in a row with one cell spanning the table width |
| `NOT_A_TABLE` | The whole region contains no data table (for example a contents page or running text) |

`tfoot` has a project-defined meaning here. A total or summary row belongs in `tbody`; the parser does not infer the meaning of arbitrary HTML footer rows.

No attributes other than `colspan`, `rowspan`, `data-level` and `data-unclear` are used. There is no `<br>`, styling, or empty spacer row.

Because the title is a single Markdown line, the boundary between a title and its units line is not preserved. Keeping them apart is a candidate for the next contract revision.

## Transcription conventions

These conventions define the training targets. They describe how printed content is transcribed, not a recovery of exact page typography; the source image and raw response are kept alongside the result.

The transcription reproduces what is printed. A title, column headings, indentation or footnote lines are written when they are visible on the page and left out when they are not; nothing is supplied from the table's content or from what such tables usually contain. Interpreting the reproduction, such as reading the first column as row labels and building row paths, is the job of the JSON conversion. Two kinds of convention go beyond appearance and are applied the same way everywhere: reading conventions (joining hyphenated words, removing dot leaders, copying values exactly), and footnote markers, which are identified by what they do rather than by how they are printed.

A table that prints neither a title nor column headings is transcribed as a body alone:

```html
<table>
  <tbody>
    <tr><td data-level="0">Wheat</td><td>1,204</td><td>1,310</td></tr>
    <tr><td data-level="0">Barley</td><td>388</td><td>-</td></tr>
  </tbody>
</table>
```

Title:

- The `# Title` line holds the text printed above the table that names it: table number, title and units line, in printed order.
- A table with no printed title has no heading line. A title is never composed from the table's content.

Structure:

- Every row has the same number of columns once spans are expanded.
- `thead` holds the rows printed as column headings: above the data, and set apart by position, rules or typeface. A table that prints no column headings has no `thead`, and no header row is invented.
- When there is a `thead`, it covers every column; the cell above the first column is a `th` and may be empty.
- Two independent tables in one image, side by side or one above the other, are two `<table>` elements in reading order (left before right, top before bottom). A heading printed above only one of them becomes that table's first header row, spanning all its columns.
- Text belonging to a previous table, such as its footnotes or a `Source:` line above this table's title, is left out.

First column:

- The first cell of each body row carries `data-level`: its visible indentation step relative to the leftmost text in that column, `0` for flush left. This applies whether or not the first column holds labels.
- A label that wraps over several lines is joined into one cell with single spaces; one label is never split into several rows.
- Dot leaders are removed, including trailing dots after the label text.
- A word hyphenated only by a line break is joined (`Acquit-` / `tals` becomes `Acquittals`); hyphens that are part of the text are kept.
- A heading row that has a label but no values keeps its own row, with empty value cells.
- Ditto marks in the label column are written as `"`.

Values:

- Values are copied exactly as printed: digits, commas, decimal points, `$`, signs, `%`. Nothing is computed, rounded, reformatted or corrected.
- An empty position is an empty cell. Printed placeholders are kept as printed: a dash is `-`, an ellipsis is `...`, an asterisk is `*`.
- A footnote marker such as `1/`, `a/` or `*` attached to a value, label, header or title is written as `<sup>1</sup>`, `<sup>a</sup>` or `<sup>*</sup>`, after the text it belongs to. This holds whether the marker is printed raised or on the line: a raised `32` after `Providers` that refers to a note is `Providers<sup>32</sup>`, and so is an inline `Providers 32/`. `<sup>` is used for nothing else.
- Illegible characters get a best reading, and the cell gets `data-unclear="1"`.

Footnotes:

- Footnotes are the lines printed below the table that belong to it: numbered or lettered notes, and `Note:` and `Source:` lines. A table with none has no `tfoot`. Text below that belongs to the next table or to the page is left out.
- Each footnote line is one `tfoot` row with a single cell spanning all columns, with its text as printed, including its printed label (`1/ Includes Newfoundland.`).

## Parser behavior

The parser removes surrounding whitespace, recognizes the sentinel, and splits the response into table fragments, each with its immediately preceding title.

These produce a blocking contract issue: an empty response, a pipe table, a `<caption>` element, prose or code fences outside the allowed shape, and a response that mixes `NOT_A_TABLE` with tables. Candidates that can still be recovered may be kept for inspection, but they are never accepted under an alternative dialect.

A `<table>` that is never closed runs to the end of the response and produces a blocking `unterminated_table` issue. A table nested inside another table makes its fragment unusable.

Some deviations from this contract are tolerated, each with a non-blocking issue, because they lose no content:

- A title written with another heading level (`## Title`), or several heading lines before one table: the lines are joined with a space in source order (`title_format`).
- `<br>` inside a cell is read as a space; other inline elements such as `<em>` keep their text (`unexpected_markup`).
- Attributes outside the contract, such as `style`, are ignored; a `data-unclear` value other than `1` still marks the cell unclear (`unexpected_markup`).
- A `th` in `tbody` is read as a body cell, and a `tr` outside `thead`, `tbody` and `tfoot` as a body row (`unexpected_markup`).
- A `data-level` that is not a nonnegative integer is dropped (`invalid_level`).

Text inside the table structure but outside any cell, such as a `<sup>` written between two cells, would be lost and is a blocking `unexpected_content` issue, like text outside tables.

A malformed fragment does not invalidate unrelated fragments when their boundaries are unambiguous. Significant repairs made by the HTML parser are recorded as issues rather than treating the repaired structure as known to be correct. The [table JSON format](table-format.md#mapping-from-the-fixed-ocr-contract) describes how the accepted constructs map to JSON, including how note labels are separated.

## Legacy annotations

Older annotation files that put the title in `<caption>` are converted once, during dataset preparation, into this format: the caption becomes the `# Title` line before its table. The originals are kept for evaluation and provenance.
