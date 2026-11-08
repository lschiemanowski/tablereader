# Table reader

Turns tables in scanned PDFs into structured JSON: a layout detector proposes regions, a fine-tuned LightOnOCR-3-4B model transcribes them into a fixed HTML contract, and deterministic code converts the transcription into table JSON. Successor to `~/workspace/ttr` and `~/workspace/treasury_ver0`, whose code and data are reference material (read-only).

## Where things are

- `docs/table-format.md`, `packages/table_core/src/table_core/table.schema.json`, `examples/`: the table JSON format (draft 0.3). This is the contract; change the spec, schema and examples together.
- `docs/transcription-format.md`: the fixed OCR transcription the model emits and the parser accepts (contract revision 1). Changing it means retraining.
- `planning/`: architecture and build plan. **Not in git** (ignored). Read `planning/architecture.md` and `planning/build-plan.md` before starting a build step; never move planning documents into tracked folders.
- `packages/table_core/`: parsing, JSON proposal, validation and codecs. Pure Python; no GPU, network, PDF or web dependencies.
- `packages/document_prep/`: page rendering (pypdfium2, pinned), crop extension over titles and footnotes, Heron layout detection (optional `layout` extra). Crop pixels depend on these versions and on `PreparationSettings`; change them deliberately.
- `scripts/`: one-off runs against table_benchmark (read-only there); outputs go to the gitignored `data/`. Run them from this repository, never from inside table_benchmark.
- Later members (`extraction`, `training`, `evaluation`, `apps/serving`) are added when their build step starts, not before.

## Commands

```bash
uv sync                    # install the workspace and dev tools
uv sync --group layout     # also torch/transformers for Heron layout detection (GPU)
uv run pytest              # tests
uv run ruff check          # lint
uv run ruff format         # format
uv run pyright             # type check (strict for table_core/src)
```

## Conventions

- Python 3.12 (matches ttr, Docling and the training stack). uv workspace; each package has its own `pyproject.toml` under `packages/`.
- Printed values stay strings; never infer numbers, units or semantic roles in `table_core`.
- Functions for deterministic transformations; Protocols only at external boundaries (detector, recognizer, job store). No plugin registries or output-format selectors.
- Expected content problems are `Issue` records; operational failures raise; nothing becomes an empty successful table.
- Each build step lands as one PR with tests.
