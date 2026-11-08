"""The Codex recognizer, against a fake CLI; and the labeling prompt against the transcription spec."""

import asyncio
import os
import stat
import sys
from pathlib import Path

import pytest
from PIL import Image

from document_prep.models import Box, DocumentSource, PreparationSettings, PreparedRegion, TableRegion
from extraction.codex import PROMPT, CodexRecognizer
from extraction.results import ComponentFailure, FinishReason, OcrResult

ROOT = Path(__file__).resolve().parents[3]
BOX = Box(left=0, top=0, right=10, bottom=10)

FAKE_CLI = f"""#!{sys.executable}
import json, os, sys, time
args = sys.argv[1:]
answer = args[args.index("--output-last-message") + 1]
image = args[args.index("--image") + 1]
prompt = sys.stdin.read()
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
usage = {{"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 80}}
if mode == "timeout":
    time.sleep(30)
if mode == "error":
    print(json.dumps({{"type": "turn.failed", "error": {{"message": "usage limit reached"}}}}))
    sys.exit(1)
if mode == "malformed":
    open(answer, "w").write("not json")
elif mode == "env":
    open(answer, "w").write(json.dumps({{"transcription": "key" if "OPENAI_API_KEY" in os.environ else "no key"}}))
else:
    check = "prompt ok" if "NOT_A_TABLE" in prompt and os.path.exists(image) else "bad call"
    open(answer, "w").write(json.dumps({{"transcription": check}}))
print(json.dumps({{"type": "thread.started"}}))
print(json.dumps({{"type": "turn.completed", "usage": usage}}))
"""


@pytest.fixture
def cli(tmp_path: Path) -> Path:
    path = tmp_path / "codex"
    path.write_text(FAKE_CLI)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def region() -> PreparedRegion:
    return PreparedRegion(
        source=DocumentSource(path=Path("x.pdf"), sha256="0" * 64),
        region=TableRegion(region_id="r", page=1, box=BOX),
        page_size=(10.0, 10.0),
        page_pixels=(10, 10),
        detected_px=BOX,
        crop_px=BOX,
        crop_box=BOX,
        image=Image.new("RGB", (10, 10), "white"),
        settings=PreparationSettings(),
        renderer="test",
    )


def recognize(cli: Path, mode: str, monkeypatch: pytest.MonkeyPatch, timeout: float = 600.0) -> OcrResult:
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    return asyncio.run(CodexRecognizer(cli=cli, timeout=timeout).recognize(region()))


def test_answer_usage_and_settings(cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = recognize(cli, "ok", monkeypatch)
    assert result.text == "prompt ok"
    assert result.finish_reason is FinishReason.STOP
    assert result.model == "gpt-6.1-terra"
    assert result.usage == {"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 80}
    assert result.settings["reasoning"] == "low"
    assert len(result.settings["prompt_sha256"]) == 64


def test_api_keys_are_not_passed_on(cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert recognize(cli, "env", monkeypatch).text == "no key"


@pytest.mark.parametrize(("mode", "code"), [("error", "codex_failed"), ("malformed", "malformed_answer")])
def test_failures_are_retryable_component_failures(
    cli: Path, monkeypatch: pytest.MonkeyPatch, mode: str, code: str
) -> None:
    with pytest.raises(ComponentFailure) as caught:
        recognize(cli, mode, monkeypatch)
    assert (caught.value.code, caught.value.retryable) == (code, True)
    if mode == "error":
        assert "usage limit reached" in caught.value.message


def test_timeout(cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ComponentFailure) as caught:
        recognize(cli, "timeout", monkeypatch, timeout=0.5)
    assert caught.value.code == "timeout"


def test_command_is_locked_down(tmp_path: Path) -> None:
    command = CodexRecognizer().command(tmp_path / "i.png", tmp_path / "s.json", tmp_path / "a.json")
    joined = " ".join(command)
    for part in ("--sandbox read-only", "--disable shell_tool", "--ignore-user-config", "--ephemeral", "never"):
        assert part in joined
    assert 'web_search="disabled"' in command


def test_prompt_carries_the_current_transcription_spec() -> None:
    doc = (ROOT / "docs/transcription-format.md").read_text(encoding="utf-8")
    spec = doc[doc.index("## Response shape") : doc.index("## Parser behavior")].rstrip() + "\n"
    assert PROMPT.endswith(spec), "rebuild prompts/transcription.md from docs/transcription-format.md"
    assert "Contract revision 1" in doc and "contract revision 1" in PROMPT


def test_environment_is_not_mutated() -> None:
    before = dict(os.environ)
    CodexRecognizer()
    assert dict(os.environ) == before
