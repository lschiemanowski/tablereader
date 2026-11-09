"""Evaluation: the record metric on table JSON, and grid comparison against cell-grid references (table_benchmark v1).

The record metric (version `RECORDS_VERSION`) scores a crop's extracted tables against its reference tables as
the overlap of `(row path, column path, value)` records. Grid comparison turns a transcription back into a cell
grid (title and footnotes dropped, markers reinserted) to cross-check labels against v1's grids.
"""
