"""Label acceptance, training targets, recrop loading and the evaluation's crop tables, without a GPU."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from document_prep.models import PreparationSettings
from evaluation.records import NOT_A_TABLE, crop_score
from extraction.extract import interpret
from extraction.results import FinishReason, OcrResult, encode_region_result
from training.data import Recrops, reinterpret, rejection, training_target
from training.pipeline import crop_tables

TABLE = '# Receipts\n<table>\n  <tbody>\n    <tr><td data-level="0">Taxes</td><td>100</td></tr>\n  </tbody>\n</table>'
UNCOVERED = (  # the second row leaves a body position uncovered: blocking, so unresolved
    '<table><tbody><tr><td data-level="0">a</td><td>1</td><td>2</td></tr><tr><td data-level="0">b</td></tr>'
    "</tbody></table>"
)


def label(text: str, finish: FinishReason = FinishReason.STOP, completion_tokens: int = 10) -> dict[str, Any]:
    response = OcrResult(text=text, model="m", finish_reason=finish, usage={"completion_tokens": completion_tokens})
    return {"region_id": "r-0", "split": "train", **encode_region_result(interpret("r-0", response))}


def test_ready_tables_and_not_a_table_are_accepted() -> None:
    assert rejection(label(TABLE)) is None
    assert rejection(label("NOT_A_TABLE")) is None


@pytest.mark.parametrize(
    ("text", "finish", "reason"),
    [
        (TABLE, FinishReason.LENGTH, "cut off"),
        ("no table here", FinishReason.STOP, "failed"),
        (UNCOVERED, FinishReason.STOP, "not ready"),
    ],
)
def test_rejections(text: str, finish: FinishReason, reason: str) -> None:
    assert rejection(label(text, finish)) == reason


def test_self_label_token_limit() -> None:
    long = label(TABLE, completion_tokens=12000)
    assert rejection(long) is None
    assert rejection(long, max_tokens=10000) == "too long"


def test_training_target_drops_indentation_only() -> None:
    target = training_target("\n" + TABLE + "\n")
    assert (
        target == '# Receipts\n<table>\n<tbody>\n<tr><td data-level="0">Taxes</td><td>100</td></tr>\n</tbody>\n</table>'
    )
    assert rejection(label(target)) is None


def test_reinterpret_reproduces_the_saved_result() -> None:
    saved = label(TABLE)
    assert encode_region_result(reinterpret(saved)) == {k: v for k, v in saved.items() if k not in {"split"}} | {
        "region_id": "r-0"
    }


def test_crop_tables_for_scoring() -> None:
    reference = crop_tables(reinterpret(label(TABLE)))
    assert crop_score(reference, crop_tables(reinterpret(label(training_target(TABLE))))) == 1.0
    assert crop_tables(reinterpret(label("NOT_A_TABLE"))) is NOT_A_TABLE
    assert crop_tables(reinterpret(label("no table here"))) is None


def test_recrops_rebuild_prepared_regions(tmp_path: Path) -> None:
    settings = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(PreparationSettings()).items()}
    (tmp_path / "settings.json").write_text(json.dumps({"settings": settings, "renderer": "pdfium test"}))
    (tmp_path / "images").mkdir()
    Image.new("L", (40, 20), 255).save(tmp_path / "images/abc-0.png")
    row = {
        "table_id": "doc:p0001:r01", "part": 0, "split": "train", "document_id": "doc", "page": 1, "rotation_cw": 0,
        "detected_box": [1, 2, 3, 4], "crop_box": [0, 1, 4, 5], "page_size": [10, 10], "detected_px": [4, 8, 12, 16],
        "crop_px": [0, 4, 16, 20], "page_pixels": [40, 40], "image": "images/abc-0.png", "sha256": "x",
    }  # fmt: skip
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    recrops = Recrops.load(tmp_path)
    assert recrops.settings == PreparationSettings()
    (region,) = recrops.regions(recrops.rows)
    assert (region.region.region_id, region.image.size, region.renderer) == ("abc-0", (40, 20), "pdfium test")
