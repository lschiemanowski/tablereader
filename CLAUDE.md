# Table reader

Turns tables in scanned PDFs into structured JSON: a layout detector proposes regions, a fine-tuned LightOnOCR-3-4B model transcribes them into a fixed HTML contract, and deterministic code converts the transcription into table JSON. Successor to the local `ttr` and `treasury_ver0` projects (not public), whose code and data are reference material (read-only).

## Where things are

- `docs/table-format.md`, `packages/table_core/src/table_core/table.schema.json`, `examples/`: the table JSON format (draft 0.3). This is the contract; change the spec, schema and examples together.
- `docs/transcription-format.md`: the fixed OCR transcription the model emits and the parser accepts (contract revision 1). Changing it means retraining.
- `planning/`: architecture and build plan. **Not in git** (ignored). Read `planning/architecture.md` and `planning/build-plan.md` before starting a build step; never move planning documents into tracked folders.
- `packages/table_core/`: parsing, JSON proposal, validation and codecs. Pure Python; no GPU, network, PDF or web dependencies.
- `packages/document_prep/`: page rendering (pypdfium2, pinned), crop extension over titles and footnotes, Heron layout detection (optional `layout` extra). Crop pixels depend on these versions and on `PreparationSettings`; change them deliberately.
- `scripts/`: one-off runs against table_benchmark (read-only there); outputs go to the gitignored `data/`. Run them from this repository, never from inside table_benchmark.
- `packages/extraction/`: the `TableRecognizer` protocol, result types, bounded concurrent extraction over prepared regions, and `CodexRecognizer` (GPT-6.1 Terra via headless `codex exec`, used for labeling). Its prompt copies sections of `docs/transcription-format.md` verbatim; a test enforces it, so edit both together.
- `packages/evaluation/`: the record metric (`records-1`; bump the version with any change to normalization or records) and cell-grid comparison with table_benchmark references.
- `packages/training/`: label acceptance, LoRA fine-tuning (`training.sft`) and the self-training pipeline (`training.pipeline`). GPU dependencies come with `uv sync --group train`; run long jobs in a separate environment (`UV_PROJECT_ENVIRONMENT=.venv-train`) so a plain `uv run` cannot uninstall torch under them.
- `packages/tablereader/`: the public interface. Model release manifests (`release.json`), `read_pdf`/`TableReader`, outputs, the `tablereader` CLI and the FastAPI service with its file job store. vLLM is a separate tool install.

## Commands

```bash
uv sync                    # install the workspace and dev tools
uv sync --group layout     # also torch/transformers for Heron layout detection (GPU)
uv sync --group gpu        # everything the reader and service need on a GPU (plus `uv tool install vllm`)
uv sync --group train      # fine-tuning dependencies
uv run pytest              # tests
uv run ruff check          # lint
uv run ruff format         # format
uv run pyright             # type check (strict for most package sources)
```

## Conventions

- Python 3.12 (matches ttr, Docling and the training stack). uv workspace; each package has its own `pyproject.toml` under `packages/`.
- Printed values stay strings; never infer numbers, units or semantic roles in `table_core`.
- Functions for deterministic transformations; Protocols only at external boundaries (detector, recognizer, job store). No plugin registries or output-format selectors.
- Expected content problems are `Issue` records; operational failures raise; nothing becomes an empty successful table.
- Each build step lands as one PR with tests.
