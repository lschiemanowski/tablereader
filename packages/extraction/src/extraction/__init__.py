"""Extraction: recognize prepared table regions and turn each response into checked table results.

`extract_regions` runs a `TableRecognizer` over regions with bounded concurrency and yields `RegionResult`s in
source order; `interpret` is the deterministic part (parse, propose, validate). Recognizers: `CodexRecognizer`
(an OpenAI model through the Codex CLI, for labeling) and `SavedResponses` (for tests and re-evaluation).
"""
