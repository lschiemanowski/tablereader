# tablereader

Reads the tables in scanned and born-digital PDFs into structured JSON.

1. **Detect**: [Docling Heron](https://huggingface.co/docling-project/docling-layout-heron) finds the tables on
   each page.
2. **Crop**: each table is cut out at 300 DPI and extended over its title, units line and footnotes, stopping at
   running text.
3. **Transcribe**: a fine-tuned [LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B), served
   with vLLM, writes what is printed in a fixed HTML format
   ([docs/transcription-format.md](docs/transcription-format.md)): title, column headings, row indentation,
   footnote markers and footnotes.
4. **Convert**: deterministic code turns the transcription into table JSON
   ([docs/table-format.md](docs/table-format.md)) with column headers, row labels and their hierarchy, cells
   and footnote references, and reports what it could not resolve. Printed values stay strings.

## Status

Version 0.1, first release. The model
[`lschiemanowski/tablereader-lightonocr-v0`](https://huggingface.co/lschiemanowski/tablereader-lightonocr-v0)
was fine-tuned on 995 transcriptions of tables from 107 publication series: UK public-sector reports, US federal
and historical statistics, German budget annexes and others. Scored on 1,006 validation crops from different
series than the training tables, as the share of printed values whose row path, column path and text match the reference exactly:

| | All | Born-digital | Scanned |
| --- | ---: | ---: | ---: |
| tablereader-lightonocr-v0 | 0.797 | 0.835 | 0.511 |

72% of validation crops are read without a single difference. Known limits:

- Scans read much worse than born-digital pages.
- The model does not yet answer "not a table" for a region that holds none; it transcribes it anyway.
- Tables printed sideways are not detected as sideways, and a table continued on the next page is read as two
  tables.
- The references are model transcriptions (GPT-6.1 Terra) checked by parsing, not by a person.

The training and evaluation data is the [`lschiemanowski/scannedtables`](https://huggingface.co/datasets/lschiemanowski/scannedtables)
dataset.

## Install

Needs Linux, an NVIDIA GPU (developed on a 32 GB RTX 5090; smaller GPUs are untested), Python 3.12 and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lschiemanowski/tablereader
cd tablereader
uv sync --group gpu             # the reader, torch for table detection
uv tool install vllm==0.33.0    # serves the model; a separate environment
```

The model (8.5 GB) and the detector are downloaded from Hugging Face on first use.

## Use

### Command line

```bash
uv run tablereader read report.pdf --pages 13-20 --out out/report
```

writes into `out/report`:

| File | Contents |
| --- | --- |
| `tables.jsonl` | one table per line: ids, page, outcome (`ready`, `unresolved`, `failed`), issues and the table JSON |
| `document.json` | every detected region with its boxes (PDF points), status, raw transcription and tables |
| `crops/` | the crop images the model read |
| `review.html` | each crop beside the tables read from it, to check by eye |

### Python

```python
from tablereader import TableReader, read_pdf

result = read_pdf("report.pdf", pages="13-20")
for table in result.tables:
    print(table.table_id, table.outcome, table.candidate.title if table.candidate else None)

with TableReader() as reader:  # keeps the model loaded between documents
    for path in paths:
        result = reader.read(path)
```

`table.candidate` is a `table_core.models.Table`; `table_core.codec.encode_table` gives its JSON.

### HTTP service

```bash
uv run tablereader serve --jobs jobs/ --port 8080
```

Reading a PDF takes from seconds to minutes, so the service works with jobs. Jobs run one at a time on the GPU
and are kept on disk, so a restart resumes them.

```bash
curl -F file=@report.pdf -F pages=13-20 -H "Idempotency-Key: report-1" localhost:8080/v1/jobs   # 202, the job
curl localhost:8080/v1/jobs/<job_id>             # state and progress
curl localhost:8080/v1/jobs/<job_id>/tables      # the tables, once the job has succeeded
```

| Endpoint | |
| --- | --- |
| `POST /v1/jobs` | upload a PDF (multipart `file`, optional `pages`, optional `Idempotency-Key` header) |
| `GET /v1/jobs`, `GET /v1/jobs/{id}` | jobs, state (`queued`, `running`, `succeeded`, `failed`, `canceled`) and progress |
| `GET /v1/jobs/{id}/tables` | one object per table, as in `tables.jsonl` |
| `GET /v1/jobs/{id}/document` | the full result, as in `document.json` |
| `GET /v1/jobs/{id}/review` | the review page |
| `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}` | cancel a job; delete a finished one |
| `GET /healthz` | service and model status |

The service has no authentication. Run it on a trusted network or behind a proxy that adds it.

## Repository

| Path | |
| --- | --- |
| `docs/` | the table JSON format (draft 0.3) and the transcription format (contract revision 1) |
| `packages/table_core` | parsing, conversion to table JSON, validation; pure Python |
| `packages/document_prep` | rendering, table detection, cropping |
| `packages/extraction` | recognizers (vLLM, Codex for labeling) and concurrent extraction |
| `packages/evaluation` | the record metric and grid comparison |
| `packages/training` | label acceptance, LoRA fine-tuning, self-training |
| `packages/tablereader` | model releases, `read_pdf`, CLI and HTTP service |
| `scripts/` | the runs that built the dataset from a local benchmark collection |

Development:

```bash
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check && uv run pyright
```

## License

Code: Apache 2.0. The model inherits LightOnOCR-3-4B's Apache 2.0 license. The dataset's annotations are
CC BY 4.0, and its images keep their source documents' terms.
