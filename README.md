# tablereader

Reads the tables in PDFs, including scans, into structured JSON. It finds the tables on each page, crops each one
together with its title and footnotes, transcribes it with a fine-tuned
[LightOnOCR-3-4B](https://huggingface.co/lschiemanowski/tablereader-lightonocr-v0), and converts the
transcription into JSON with headers, row labels, cells and footnotes ([format](docs/table-format.md)).

## Usage

Needs Linux, an NVIDIA GPU, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lschiemanowski/tablereader
cd tablereader
uv sync --group gpu
uv tool install vllm==0.33.0
```

Command line:

```bash
uv run tablereader read report.pdf --pages 13-20 --out out/report
```

This writes `tables.jsonl` (one table per line), `document.json`, the crops, and `review.html`, which shows each
crop beside the table read from it.

Python:

```python
from tablereader import read_pdf

result = read_pdf("report.pdf", pages="13-20")
for table in result.tables:
    print(table.table_id, table.outcome, table.candidate)
```

HTTP service:

```bash
uv run tablereader serve --port 8080
curl -F file=@report.pdf -F pages=13-20 localhost:8080/v1/jobs   # returns a job id
curl localhost:8080/v1/jobs/<job_id>                            # state and progress
curl localhost:8080/v1/jobs/<job_id>/tables                     # the tables, when done
```
