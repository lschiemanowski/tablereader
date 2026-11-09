"""Model releases (planning/architecture.md, "Training and model releases").

A release is a merged model directory with a `release.json` manifest: the contract revision the model emits,
the preparation settings and renderer its training crops came from, the table detector, and the generation
settings. Inference reads the manifest, so a release is run the way it was trained. A release is a local
directory or a Hugging Face model repository holding one.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_prep.detect import TABLE_THRESHOLD
from document_prep.heron import HERON_REPO, HERON_REVISION
from document_prep.models import PreparationSettings
from document_prep.prepare import RENDERER
from extraction.vllm import MAX_EDGE, MAX_TOKENS

CONTRACT_REVISION = 1
MANIFEST = "release.json"
DEFAULT_RELEASE = "lschiemanowski/tablereader-lightonocr-v0"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRelease:
    release_id: str
    model_dir: Path
    contract_revision: int
    preparation: PreparationSettings
    renderer: str
    layout_model: str
    table_threshold: float
    max_edge: int
    max_tokens: int
    temperature: float
    enable_thinking: bool
    base_model: str
    training: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        payload = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "model_dir"}
        payload["preparation"] = dataclasses.asdict(self.preparation)
        return payload


def _settings(stored: dict[str, Any]) -> PreparationSettings:
    fields: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in stored.items()}  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return PreparationSettings(**fields)


def load_release(source: str | Path) -> ModelRelease:
    """The release in directory `source`, or in Hugging Face repository `source` (downloaded once, then cached)."""
    directory = Path(source)
    if not directory.is_dir():
        from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

        directory = Path(snapshot_download(repo_id=str(source)))  # pyright: ignore[reportUnknownArgumentType]
    manifest = directory / MANIFEST
    if not manifest.exists():
        raise ValueError(f"{source} has no {MANIFEST}; not a tablereader model release")
    data: dict[str, Any] = json.loads(manifest.read_text())
    if data.get("contract_revision") != CONTRACT_REVISION:
        found = data.get("contract_revision")
        raise ValueError(f"{source} emits contract revision {found}; this tablereader reads {CONTRACT_REVISION}")
    return ModelRelease(
        release_id=data["release_id"],
        model_dir=directory,
        contract_revision=data["contract_revision"],
        preparation=_settings(data["preparation"]),
        renderer=data["renderer"],
        layout_model=data["layout_model"],
        table_threshold=data["table_threshold"],
        max_edge=data["max_edge"],
        max_tokens=data["max_tokens"],
        temperature=data["temperature"],
        enable_thinking=data["enable_thinking"],
        base_model=data["base_model"],
        training=data.get("training", {}),
    )


def write_release(
    model_dir: Path, *, release_id: str, preparation: PreparationSettings, base_model: str, training: dict[str, Any]
) -> ModelRelease:
    """Write the manifest for a merged model trained on crops prepared with `preparation`."""
    release = ModelRelease(
        release_id=release_id,
        model_dir=model_dir,
        contract_revision=CONTRACT_REVISION,
        preparation=preparation,
        renderer=RENDERER,
        layout_model=f"{HERON_REPO}@{HERON_REVISION}",
        table_threshold=TABLE_THRESHOLD,
        max_edge=MAX_EDGE,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        enable_thinking=False,
        base_model=base_model,
        training=training,
    )
    (model_dir / MANIFEST).write_text(json.dumps(release.to_json(), indent=2) + "\n")
    return release
