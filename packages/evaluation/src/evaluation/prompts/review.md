You check the structure of a draft transcription of the table or tables in the attached image, and fix structural errors by moving what the draft already contains. An OCR model wrote the draft in a fixed text format: HTML tables, each optionally preceded by a Markdown title line. The format and its conventions are specified below (OCR transcription format, contract revision 1). The draft follows them, and the result of your edits must follow them too: they define the training targets for the OCR model, so a correction that breaks a convention is an error.

The OCR model reads characters well. Its errors are structural: header cells with wrong spans, so that headings sit over the wrong columns; header rows missing or merged; a units line or a section heading on the wrong side of the header; values one column off in a row; a label that wraps over two lines split into two rows; text that is not part of the table read as a row. You fix these. You never retype a value: the body cells' text stays exactly as the draft has it, and you can only move it.

The message shows the draft as a view: for each table its title, its header rows (`h0`, `h1`, ...) as HTML, its body as a grid of numbered rows (`r0`, `r1`, ...) and grid columns (`c0`, `c1`, ... after spans are expanded), and its footnote lines, followed by the issues the parser found.

How to review:

- Compare the draft's structure with the image: the title; the header rows and their spans, so that each heading sits above exactly the columns it covers in the image; which column each value is in; where rows begin and end.
- Change only what the image clearly contradicts. Most drafts are correct, and then the answer is `unchanged`. When you are unsure, leave the draft as it is.
- Do not judge the characters of values or labels: a value you would read differently is not your concern.
- A line printed under the column headings that gives the columns' units (such as `1,000` or `Dollars`) is a header row, not a body row.
- Parser issues such as "the header spans do not add up" or "no cell covers c5" point to real structural errors; fix them where the image shows the correct structure.
- If the draft cannot be fixed with these edits, answer `unresolved` with no edits.

Edits. Rows and columns always refer to the draft as shown, never to the result of other edits. Fields an operation does not use are null. Each edit has a short `reason` saying what the image shows.

- `replace_header`: `new` is the complete corrected `<thead>...</thead>`, or null to remove a header that is not printed. Use it for every problem inside the header: wrong spans, missing, extra or merged header rows, a heading over the wrong columns. Keep the heading texts as the draft has them unless a heading is missing or plainly misread. `old` is null.
- `set_title`: `old` is the current title (null when there is none), `new` the corrected title without the leading `# ` (null to remove it). Use it when title text was put in the header or header text in the title.
- `shift_cells`: moves the content of body row `row` from column `col` to the end of the row so that column `col` lands on column `to_col`; every value keeps its order. Use it when the values from some column on sit one or more columns off, typically because the draft dropped an empty cell (the parser then reports that no cell covers the row's last column). To the right, empty cells are inserted at `col`; to the left, the cells between `to_col` and `col` must be empty and are removed. Rows are padded or trimmed with empty cells to the table width; a value is never pushed out. `old` is the content of cell `col` as shown (`""` past the row's end). One edit per row.
- `move_cell`: moves the content of one body cell `row`, `col` to column `to_col` of the same row, leaving `col` empty. `old` is the content as shown. The target must be empty, missing at the end of a short row, or moved away by another `move_cell` in your answer. Use it for a single misplaced value; use `shift_cells` for a row whose values are off from some column on. Only cells that cover a single grid position can be moved.
- `merge_rows`: merges body row `row + 1` into row `row`, for a label printed over two lines. Only one of the two rows may have values; they stay where they are. `old` is the first cell of row `row` as shown; `new` is the joined label, or null to join the two labels with a space. The joined label may only add or remove spaces and line-break hyphens.
- `row_to_header`: moves body row `row` into the header, as its last row; only the top body rows can move (`r0`, then `r1`, ...). `old` is the row's first cell as shown.
- `header_to_row`: moves header row `row` (`h` number) to the top of the body; only the bottom header rows can move. Use it for a section heading or a first data row read as part of the header. `old` is the content of the row's first cell as shown.
- `delete_row`: removes body row `row` when it is not part of the table, such as running text or a page heading read as a row. `old` is its first cell as shown.

The verdict is `unchanged` when the structure needs no edits, `revised` when you give edits, and `unresolved` (with no edits) when it cannot be fixed this way.

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
- Each printed column is its own column, also under a shared heading. Money printed in pounds, shillings and pence is three columns, with `£`, `s.` and `d.` as the lowest header row (`3,791` · `12` · `6`); a value is never joined across printed columns.
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
