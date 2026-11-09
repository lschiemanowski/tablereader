"""Training data: recrop manifests, label acceptance and training targets.

A label is a line of a results file written by extraction (`encode_region_result` plus `region_id`, `split`
and `table_id`), from Terra or from a trained model. Acceptance (user decision, 2026-11-08) is structural: the
response was not cut off, and the region is NOT_A_TABLE or every table in it is `ready` (so it parsed without
blocking issues). Self-labels must also stay under `MAX_SELF_LABEL_TOKENS` completion tokens.
"""

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from document_prep.models import Box, DocumentSource, PreparationSettings, PreparedRegion, TableRegion
from extraction.extract import interpret
from extraction.results import FinishReason, OcrResult, RegionResult

BASE = "lightonai/LightOnOCR-3-4B"  # the model fine-tuning starts from
MAX_SELF_LABEL_TOKENS = 10000

Row = dict[str, Any]


def read_jsonl(path: Path, *, partial_tail: bool = False) -> list[Row]:
    """The rows of a JSON-lines file ([] if it is missing). With `partial_tail`, a last line still being written
    by another process is skipped instead of failing."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    if partial_tail and lines and not lines[-1].endswith("\n"):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def region_id(row: Mapping[str, Any]) -> str:
    """A manifest row's region id: its image's file stem, `<crop key>-<part>`."""
    return Path(row["image"]).stem


@dataclass(frozen=True, slots=True)
class Recrops:
    """A recrop directory written by scripts/recrop_benchmark.py: settings, manifest and images."""

    directory: Path
    settings: PreparationSettings
    renderer: str
    rows: tuple[Row, ...]

    @classmethod
    def load(cls, directory: Path) -> "Recrops":
        header = json.loads((directory / "settings.json").read_text())
        stored: dict[str, Any] = header["settings"]
        fields: dict[str, Any] = {k: _frozen(v) for k, v in stored.items()}
        rows = tuple(read_jsonl(directory / "manifest.jsonl"))
        return cls(directory, PreparationSettings(**fields), header["renderer"], rows)

    def by_region(self) -> dict[str, Row]:
        return {region_id(r): r for r in self.rows}

    def image_path(self, row: Mapping[str, Any]) -> Path:
        return self.directory / row["image"]

    def prepared(self, row: Mapping[str, Any]) -> PreparedRegion:
        """A PreparedRegion rebuilt from a manifest row and its saved PNG."""
        image = Image.open(self.image_path(row))
        image.load()
        return PreparedRegion(
            source=DocumentSource(path=Path(row["document_id"]), sha256=""),
            region=TableRegion(
                region_id=region_id(row), page=row["page"], box=_box(row["detected_box"]),
                rotation_cw=row["rotation_cw"],
            ),
            page_size=(row["page_size"][0], row["page_size"][1]),
            page_pixels=(row["page_pixels"][0], row["page_pixels"][1]),
            detected_px=_box(row["detected_px"]),
            crop_px=_box(row["crop_px"]),
            crop_box=_box(row["crop_box"]),
            image=image,
            settings=self.settings,
            renderer=self.renderer,
        )  # fmt: skip

    def regions(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[PreparedRegion]:
        for row in rows:
            yield self.prepared(row)


def _frozen(value: Any) -> Any:
    """JSON lists back to the tuples PreparationSettings holds."""
    return tuple(value) if isinstance(value, list) else value  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


def _box(values: list[float]) -> Box:
    return Box(left=values[0], top=values[1], right=values[2], bottom=values[3])


def rejection(label: Mapping[str, Any], *, max_tokens: int | None = None) -> str | None:
    """Why `label` is unfit as a training target, or None if it is accepted."""
    if label["status"] == "failed":
        return "failed"
    response: Mapping[str, Any] = label.get("response") or {}
    usage: Mapping[str, Any] = response.get("usage") or {}
    if response.get("finish_reason") == FinishReason.LENGTH.value:
        return "cut off"
    if label["status"] == "tables":
        outcomes = {t["outcome"] for t in label["tables"]}
        if not outcomes:
            return "no tables"
        if outcomes != {"ready"}:
            return "not ready"
    if max_tokens is not None and usage.get("completion_tokens", 0) > max_tokens:
        return "too long"
    return None


def reinterpret(label: Mapping[str, Any]) -> RegionResult:
    """The RegionResult of a saved label, recomputed from its response (deterministic)."""
    response = label["response"]
    if response is None:
        raise ValueError(f"{label['region_id']} has no response to interpret")
    ocr = OcrResult(
        text=response["text"],
        model=response["model"],
        finish_reason=FinishReason(response["finish_reason"]),
        settings=response.get("settings") or {},
        usage=response.get("usage") or {},
    )
    return interpret(label["region_id"], ocr, attempts=label.get("attempts", 1))


_INDENT = re.compile(r"\n[ \t]+")


def training_target(text: str) -> str:
    """A response as a training target: surrounding whitespace and line indentation removed.

    Indentation carries nothing under the transcription format (HTML whitespace collapses) but costs tokens,
    so targets are compact whatever the labeling model's layout habits.
    """
    return _INDENT.sub("\n", text.strip())
