---
pretty_name: Table reader benchmark
license: other
license_name: mixed-source-terms
license_link: LICENSE
task_categories:
- image-to-text
tags:
- ocr
- table-recognition
- document-understanding
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
---

# Table reader benchmark

Crops of printed tables from {documents} public-sector, scientific and historical documents, scanned and
born-digital, with a transcription of the table as HTML under a fixed format and the table as structured
JSON. Validation and test are fully transcribed; most of train is not yet. Schema version {schema_version}.

| Split | Crop parts | Accepted transcriptions | Rejected | Unlabeled |
| --- | ---: | ---: | ---: | ---: |
{split_rows}

## Where the crops come from

The crops are the table regions of the frozen v1 splits of a local table OCR benchmark built from {documents}
source PDFs. The splits are disjoint by publication series. Each region was recropped by the table reader's
document preparation: pages rendered at 300 DPI with pdfium, the detected table box extended upward over the
table's title and units line and downward over its footnotes, with layout detection (Docling Heron) stopping
the extension at running text. Tables printed sideways are rotated upright. A table split across pages has one
row per part.

{held} crop parts are held out. A crop-level rights and privacy review held named compensation, pension and
reimbursement figures, personal declarations, signatures, contractor-authored or third-party material, and
components with conflicting or unresolved reuse terms. A second screen of the transcriptions held every table
that lists named individuals (boards, committees, ministers, fees, donors, related parties) and checked each
mention of copyright, permissions or personal identifiers in its context. Both screens work mainly from text
and can miss material that neither caught; please report anything that should not be here.

## Labels

- `transcription` reproduces what is printed, in the format described in
  [transcription-format.md](transcription-format.md): a `# Title` line when a title is printed, an HTML table
  with `thead` for printed column headings, `data-level` for the printed indentation of row labels, `<sup>` for
  footnote markers and `tfoot` for footnote lines. A crop without a table is `NOT_A_TABLE`.
- `tables` is the JSON (format 0.3, [table-format.md](table-format.md), [table.schema.json](table.schema.json))
  that the table reader's deterministic converter proposes from the transcription. Printed values stay strings.
- `transcription_model` names the model that wrote the transcription. Current labels were written by
  GPT-6.1 Terra and have not been reviewed by a person. `transcription_status` is `accepted` when the
  transcription parses into tables without blocking issues and was not cut off, `rejected` otherwise, and
  `unlabeled` where no transcription exists yet.
- `v1_label` is the original benchmark's classification of the crop: `table`, `wrongly_cropped` or
  `not_a_table`. It refers to the original tighter crop.

## Columns

`region_id` (crop key and part), `table_id`, `part`, `split`, `image` (PNG, upright), `image_sha256`,
`rotation_cw` (clockwise rotation applied to the page region), `document_id`, `source_id`, `medium`
(`scanned` or `born_digital`), `page` (one-based PDF page), `crop_box` and `page_size` (PDF points, before
rotation), `v1_label`, `transcription`, `transcription_status`, `transcription_model`, `region_status`
(`tables`, `not_a_table` or `failed`), `tables` (JSON text), `rights_basis`, `attribution`, `source_url`.

## Rights

Source terms vary by document; see [LICENSE](LICENSE) and [documents.jsonl](documents.jsonl), which gives each
document's rights basis, source link and attribution. Cropping and transcription are adaptations of the source
documents.

{annotations_license}
