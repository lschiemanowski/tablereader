# Table JSON format specification

Draft version 0.3. One JSON object describes one table: original transcribed text, column-header paths, rows, merged cells, indentation, uncertainty, titles, and footnotes. Values remain strings. The format adds no inferred units, numeric types, measures, or other domain meaning.

This is the application's single JSON output format. The version identifies a revision of the contract, not a selectable format. The [JSON Schema](../schemas/table.schema.json) checks field shapes; the additional structural checks below are also required. The JSON is produced from a fixed OCR transcription; [Mapping from the fixed OCR contract](#mapping-from-the-fixed-ocr-contract) describes how.

## Scope and source preservation

One table has an ordered list of leaf columns and rectangular body-cell spans. A region containing multiple tables produces several table objects, associated with that region in the extraction result envelope. `NOT_A_TABLE`, operational failures, and partially extracted candidates are result statuses, not empty table objects.

Keep the raw model response and source images alongside the JSON. The format preserves table structure and transcribed content, but not exact typography, header-row geometry, arbitrary inline formatting, nested tables, or multi-page stitching instructions. Text may already reflect the training target's conventions, such as joined wrapped labels and removed dot leaders; JSON conversion does not reconstruct the original typography.

Document/page coordinates, crop references, model settings, processing issues, and revision history belong in the result envelope. Object IDs connect that evidence to the table. Unsupported or ambiguous structures must be reported rather than silently flattened.

## Table fields

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | Yes | Exactly `"0.3"` |
| `id` | Yes | Table identifier within its containing document/result |
| `title` | No | Text block containing the title/caption, including a printed number or units line when present |
| `label_columns` | No | Number of leading leaf columns that hold row labels; positive integer, defaults to `1` |
| `headers` | Yes | Header objects stored once; may be empty |
| `columns` | Yes | Nonempty array of leaf columns in source order |
| `rows` | Yes | Body row objects in source order; each has an ID and the cells starting in that row |
| `notes` | No | Notes in source reading order; defaults to an empty array |
| `note_refs` | No | IDs of notes attached to the table as a whole; defaults to an empty array |

An absent title means no title was represented, not proof that the source had none. `rows: []` is a valid header-only table representation, not a non-table classification.

IDs are nonempty strings. Header, column, row, body-cell, and note IDs must be distinct from each other within a table. They are opaque: consumers must not derive meaning from their spelling. Preserve IDs for deterministic edits when object identity is known. A new independent extraction or full candidate rewrite may use new IDs; comparisons must not assume cross-run identity. Retain the old artifact rather than overwriting it.

Unknown properties are rejected. Omit optional fields rather than using `null`. Defaults describe consumer behavior; JSON Schema does not insert them. Consumers reject unsupported schema versions. The initial implementation supports 0.3 only.

## Text blocks and uncertainty

Titles, headers, and body cells carry `text` and may carry `markers`, `note_refs`, and `unclear`. Headers and cells also have the structural fields described below.

All text is Unicode. Preserve value characters, punctuation, dashes, and explicit empty strings. Do not parse numbers, fill blanks, expand abbreviations, or rewrite wording. Decode HTML character references; normalize CRLF/CR to LF. Discard indentation between structural HTML elements, but preserve cell text-node content. Markup such as `sup` and its content are removed from `text` and recorded as a marker at the position where they occurred, so `text` never contains marker characters. After removing markers, strip surrounding whitespace from `text`; a marker that stood in stripped leading or trailing whitespace takes position `0` or the end of the text. Compute marker positions only after these text transformations, and perform no further Unicode normalization.

`unclear: true` means the source transcription explicitly flagged uncertainty. Omit it when no uncertainty annotation was supplied. `false` means no unclear flag, not verified accuracy; consumers must not read either omission or false as a confidence guarantee. This annotation is allowed on a title, header, body cell, or note. It never authorizes guessing a replacement value.

## Headers and columns

A header has `id`, `text`, and the optional text annotations. A column has `id` and `headers`, an ordered array of header IDs from outermost to innermost:

```json
{
  "headers": [
    {"id": "h1", "text": "Revenue"},
    {"id": "h2", "text": "2025"},
    {"id": "h3", "text": "Actual"},
    {"id": "h4", "text": "Budget"}
  ],
  "columns": [
    {"id": "c1", "headers": ["h1", "h2", "h3"]},
    {"id": "c2", "headers": ["h1", "h2", "h4"]}
  ]
}
```

This is a fragment; complete table examples are linked below. Resolving `c1` yields `Revenue -> 2025 -> Actual`. These labels are transcribed strings, not semantic field types.

A header spanning columns is one object shared by their paths. A header spanning header rows appears once in a path. Distinct source headers with equal text retain distinct IDs. A column with no represented header has `headers: []`; an explicitly blank header has its own object with `text: ""`. Paths may have different lengths.

Each header must be referenced. An ID cannot repeat within one path. A shared header has the same ancestor prefix wherever it occurs, and the columns referencing it form a consecutive interval. This describes nested column-header structure; arbitrary crossing header associations are outside the initial format.

## Rows, cells, and spans

A row object has `id` and `cells`. The row ID lets issues, review results, and evaluation reports refer to a row, including one whose cells all start above it. Like other IDs, it identifies the row within one saved candidate, not across extraction runs.

| Row field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Row identity within the table |
| `cells` | Yes | Cells starting in this row, ordered by first covered column; may be empty |

| Cell field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Source-cell identity within the table |
| `columns` | Yes | Nonempty ordered list of consecutive leaf-column IDs covered by the cell |
| `text` | Yes | Transcribed content, excluding footnote-marker text |
| `rowspan` | No | Positive integer; defaults to `1` |
| `level` | No | Nonnegative indentation level on a row-label cell; zero means flush left |
| `unclear` | No | Explicit transcription-uncertainty annotation |
| `markers` | No | Inline footnote-marker occurrences; defaults to an empty array |
| `note_refs` | No | Object-level note attachments without a localized marker; defaults to an empty array |

Store a cell once in its starting row. Within a row, order cells by their first covered column. A cell starting in row index `r` with span `s` covers rows `r` through `r + s - 1`; its column list defines the other dimension. Never duplicate text into continuation positions. A row's `cells` may be empty if cells starting above it cover every position.

For example, `{"id":"a","columns":["c1","c2"],"text":"Not disclosed","rowspan":2}` is one cell covering two columns and two rows. Two separate cells with equal text remain two objects. A blank cell uses `text: ""`; an uncovered grid position is an error, not an implicit blank.

The first `label_columns` leaf columns hold row labels (the table's stub); the remaining columns hold values. This is a structural position, not a semantic role. `label_columns` must leave at least one value column unless the table has only one column.

`level` describes indentation, not a semantic role. It may only be attached to a cell covering the first column. A vertically merged label retains one level and one ID across its covered rows. Missing `level` means unknown/unrecorded, not an implicit zero. When the fixed OCR contract requires a level but none was supplied, preserve the candidate and report a non-blocking `missing_level` issue. The table can still be publishable; only the row paths that depend on the missing level are unresolved.

### Deriving row paths

A row path is derived rather than redundantly stored. First resolve spans. Marker text is already excluded from `text`, so derivation uses the stored strings directly.

A row's label is the sequence of nonempty texts visible in its label columns, in column order, including labels spanning from above. With the default single label column it is one string. Its level is the `level` of the cell covering the first column.

A heading row has a nonempty label and no nonempty text in its value columns. A dash or other printed placeholder is value text. An entirely blank row is not a heading. Keep heading rows as ordinary rows with their empty value cells.

Maintain an ordered list of open headings. For a new heading at level `n`, retain headings with levels less than `n`. For a value row at level `n`, retain headings with levels less than or equal to `n`. The path is those headings' labels followed by the current row's label, each label contributing its texts in column order. Add a new heading to the open list after constructing its path. Value rows never become parents merely due to indentation.

Thus `Receipts:` at level 0 followed by `Taxes` at level 1 produces `Receipts: -> Taxes`. A value row at level 0 can also remain under that heading until a new heading closes it. With two label columns, a merged `Germany` label above `Manufacturing` and `Services` produces `Germany -> Manufacturing` and `Germany -> Services`. If the relevant indentation or heading status is unresolved, report the row path as unresolved instead of inventing it.

The rule applies to the initial transcription convention, not to arbitrary table semantics. For hierarchy purposes, inspect value text visible in the resolved row, including a value spanning from above. When exporting value records, emit a merged value only at its starting position; resolving a row must not multiply that observation. A label cell spanning into value columns does not become its own value or establish an unambiguous heading; flag that case for inspection.

## Marker occurrences

A marker records footnote-marker text that was removed from the containing object's `text`, together with the position where it occurred:

```json
{
  "id": "v1",
  "columns": ["c1"],
  "text": "10",
  "markers": [{"at": 2, "text": "1", "note_ref": "n1"}]
}
```

This represents `10<sup>1</sup>`: the value text is `10`, and the marker text is `1`. The literal value `101` is stored as `text: "101"` with no marker. A consumer that ignores `markers` therefore reads the correct value; it loses only the footnote link.

| Marker field | Required | Meaning |
| --- | --- | --- |
| `at` | Yes | Zero-based Unicode code-point position in `text` where the marker occurred; `0` is before the first character, `len(text)` is after the last |
| `text` | Yes | The marker's printed text, such as `1`, `a`, or `*`; contains at least one non-whitespace character |
| `note_ref` | No | ID of the uniquely linked note; omit when unresolved |

Positions count Unicode code points, not UTF-8 bytes, UTF-16 code units, or visual glyphs. In JavaScript, index `Array.from(text)` rather than the raw UTF-16 string. For `A😀<sup>1</sup>`, `text` is `A😀` and the marker is at position 2.

Positions must satisfy `0 <= at <= len(text)` in code points and be non-decreasing in array order. Several markers may share a position; their array order is their printed order, so `10<sup>1</sup><sup>2</sup>` stores two markers at position 2. Repeated references to the same note are allowed. Reject unpaired Unicode surrogates. Text edits must update marker positions atomically. Do not create a marker merely because a suffix resembles a footnote number; marker identity must come from the transcription or source review.

To reconstruct the transcribed string with markers, insert each marker's text at its position, processing markers from last to first. For unresolved markers, retain the marker and omit `note_ref`. This preserves the difference between a value and its marker even when the note is absent.

## Notes and links

A note has required `id` and `text`, optional printed `label`, and optional `unclear`:

```json
{"id": "n1", "label": "1/", "text": "Provisional figure."}
```

Separate a clearly identified label from the note body. Otherwise keep the entire note in `text` and omit `label`. Unmarked general notes also omit `label`. Repeated labels are allowed and remain separate note IDs. Notes do not contain structured links to other notes in this version.

Under the fixed OCR contract, the converter separates a label only when the trimmed note line begins with one of these forms, followed by whitespace and nonempty text:

- a `<sup>` element, whose content is the label: `<sup>a</sup> Provisional figure.` gives label `a`;
- one to three ASCII letters or digits followed by `/`: `1/`, `a/`, `p/`, `12/`;
- one or more of `*`, `†`, `‡`, `§`.

The label keeps its printed spelling, and `text` is the remainder with surrounding whitespace removed. Any other line has no label, including `Source:` and `Note:` lines. The converter does not split a line that contains several printed notes; it stays one note, labeled from its start, and markers referring to the later notes remain unresolved. The transcription convention asks for one note per `tfoot` row, so such lines indicate a transcription problem.

Two kinds of link serve different purposes:

- `markers[].note_ref` attaches a particular inline occurrence to a note.
- `note_refs` attaches a note to the table, title, header, or body cell when there is no localized marker for that attachment, such as a general table note.

Do not duplicate a target in an object's `note_refs` when one of that object's markers already links to it. The effective notes for a text-bearing object are the union of both kinds of link. Header attachments stay on the header and are not copied to descendant cells. General notes explicitly attached to the table use the table's `note_refs`.

An unresolved note remains in `notes`; absence of links does not imply it applies to the whole table. A missing note does not justify a dangling reference or fabricated content. Record ambiguity in the result envelope.

For deterministic matching under the fixed OCR contract, trim surrounding whitespace and remove one trailing slash from both marker text and note label. Link only a unique match within the table. Preserve original spellings. If two notes have the same normalized label, leave the marker unlinked unless source review establishes the attachment. A valid ID reference alone does not prove that its association is correct.

## Mapping from the fixed OCR contract

The model emits an optional Markdown title immediately before each HTML table. That heading supplies `title.text`; `<caption>` is not an alternative runtime input. Legacy caption-based labels are converted offline for training/evaluation. In titles, `<sup>...</sup>` may explicitly mark an inline note marker, just as in header and body cells; a flattened number alone does not establish a marker.

Header cells become shared header objects and ordered column paths. Body `tr` elements become row objects, and body cells retain their spans, `data-level`, and `data-unclear`. Superscript contents are removed from text and become markers at their positions. Under this contract, `tfoot` contains note lines; totals remain body rows. An unclear annotation on a footnote cell becomes `note.unclear`.

The OCR contract cannot express more than one label column, so the converter omits `label_columns` and the default of 1 applies. For a table with several printed label columns, the later label columns are then exported as value columns. That is a known limitation of the current contract, not a property of the source. Reference annotations and reviewed corrections may set `label_columns` explicitly; extracting it from the model would require a contract revision.

Keep notes scoped to their table. Multiple tables in one response become separate table objects; there is no cross-table note lookup. Pipe tables, conflicting wrappers, or sentinel/table mixtures produce contract issues rather than selecting a different input format.

## Validation and publication

Conformance requires shape validation plus these structural checks:

1. IDs are unique within their specified scopes; references resolve to the correct object type.
2. Columns, rows, and cells retain source order. Header paths satisfy the prefix and consecutive-interval rules.
3. Each cell covers consecutive columns and a positive number of rows within bounds. Every body position is covered exactly once.
4. `level`, when present, is a nonnegative integer on a cell covering the first column. `label_columns` does not exceed the number of columns and leaves a value column when there is more than one column.
5. Marker positions lie within the containing text in code points and are non-decreasing; marker text contains non-whitespace; linked notes exist.
6. Each `note_refs` list contains distinct existing IDs and does not duplicate the same object's marker links. Unreferenced notes and unresolved markers are allowed.
7. Strings contain valid Unicode scalar values. Marker positions remain consistent with the final stored text.

JSON Schema checks types, fields, and local array constraints. It does not prove grid coverage, reference integrity, marker positions relative to text, or source fidelity. Schema-valid candidates can still be structurally invalid. Structural validity itself does not certify transcription accuracy.

Unresolved markers and unclear cells can coexist with an otherwise publishable table and visible issues. Structural errors prevent publication. Preserve failed/unresolved candidates in the result envelope rather than emitting them as conforming tables.

## Examples and revision notes

- [Multilevel headers and notes](../examples/multilevel-notes.json): three header levels, inline markers on a title/header/cell, and an unmarked general note.
- [Merged cells](../examples/merged-cells.json): two label columns, a vertically merged label, and a horizontally merged value with one marker link.
- [Duplicate headers and blanks](../examples/duplicate-headers.json): repeated header text, explicit blank/dash cells, and ambiguous repeated footnote labels.
- [Row hierarchy and uncertainty](../examples/row-hierarchy.json): heading rows, indented labels, a same-level child, and an unclear value.
- [Marker boundaries](../examples/marker-boundaries.json): numeric suffix versus literal digits, an unresolved marker, Unicode code-point offsets, and repeated links to one note.

Version 0.2 adds `level`, `unclear`, and marker occurrences, and defines object identity within each saved candidate. It also distinguishes marker links from unlocalized note attachments. Existing 0.1 artifacts cannot recover lost indentation or marker positions merely by changing the version field; reconstruct those annotations from the saved transcription or source evidence. This revision does not add a runtime format-selection mechanism.

Version 0.3 makes three incompatible changes. Marker text is removed from `text` and stored in the marker with an insertion position (`at`), replacing the 0.2 ranges (`start`, `end`). Body rows become objects with an `id` and `cells`. The optional `label_columns` field records how many leading columns hold row labels. A 0.2 table converts mechanically: cut each range out of `text`, store it as a marker at the range's start minus the length of the marker text removed before it, and wrap each row array in an object with a new row ID. 0.3 also makes a missing `level` a non-blocking issue and fixes the rule for separating note labels.
