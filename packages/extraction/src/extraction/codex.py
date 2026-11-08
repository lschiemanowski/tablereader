"""A recognizer that asks an OpenAI model, through the headless Codex CLI, for a transcription (used for labeling).

Each call runs `codex exec` in an empty temporary directory, read-only, with the shell tool and web search
disabled and without user configuration, and with API keys removed from the environment so the ChatGPT
sign-in is used. The prompt is `prompts/transcription.md`: an instruction followed by the transcription format's
response shape, constructs and conventions, copied from docs/transcription-format.md (a test keeps them equal).
The answer comes back as structured output, `{"transcription": "..."}`.
"""

import asyncio
import hashlib
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from document_prep.models import PreparedRegion
from extraction.results import ComponentFailure, FinishReason, OcrResult

DEFAULT_CLI = Path("/usr/lib/chatgpt/resources/codex")
PROMPT: str = files("extraction").joinpath("prompts/transcription.md").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"transcription": {"type": "string"}},
    "required": ["transcription"],
    "additionalProperties": False,
}
_REMOVED_ENV = {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"}


class CodexRecognizer:
    """Transcribes regions with `model` through `codex exec`. Operational failures raise `ComponentFailure`."""

    def __init__(
        self,
        *,
        cli: Path = DEFAULT_CLI,
        model: str = "gpt-6.1-terra",
        reasoning: str = "low",
        timeout: float = 600.0,
    ) -> None:
        self.cli = cli
        self.model = model
        self.reasoning = reasoning
        self.timeout = timeout
        self._env = {k: v for k, v in os.environ.items() if k not in _REMOVED_ENV}

    def command(self, image: Path, schema: Path, answer: Path) -> list[str]:
        return [
            str(self.cli),
            "--ask-for-approval",
            "never",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--disable",
            "shell_tool",
            "-c",
            'web_search="disabled"',
            "-c",
            f'model_reasoning_effort="{self.reasoning}"',
            "--model",
            self.model,
            "--json",
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(answer),
            "--image",
            str(image),
            "-",
        ]

    async def recognize(self, region: PreparedRegion) -> OcrResult:
        with tempfile.TemporaryDirectory(prefix="tablereader-codex-") as temp:
            directory = Path(temp)
            image, schema, answer = directory / "region.png", directory / "schema.json", directory / "answer.json"
            region.image.save(image, "PNG")
            schema.write_text(json.dumps(ANSWER_SCHEMA))
            process = await asyncio.create_subprocess_exec(
                *self.command(image, schema, answer),
                cwd=directory,
                env=self._env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(PROMPT.encode()), self.timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
                raise ComponentFailure(
                    "timeout", f"codex did not answer within {self.timeout:.0f} s", retryable=True
                ) from None
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
            events = _events(stdout.decode("utf-8", errors="replace"))
            usage = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
            if usage is None or process.returncode:
                messages = [_error_message(e) for e in events if e.get("type") in {"error", "turn.failed"}]
                detail = next((m for m in reversed(messages) if m), f"exit status {process.returncode}")
                raise ComponentFailure("codex_failed", detail[:600], retryable=True)
            try:
                text = json.loads(answer.read_text(encoding="utf-8"))["transcription"]
                if not isinstance(text, str):
                    raise TypeError("transcription is not a string")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ComponentFailure("malformed_answer", str(error), retryable=True) from error
        return OcrResult(
            text=text,
            model=self.model,
            finish_reason=FinishReason.STOP,
            settings={"via": "codex exec", "reasoning": self.reasoning, "prompt_sha256": PROMPT_SHA256},
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )


def _events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)  # pyright: ignore[reportUnknownArgumentType]
    return events


def _error_message(event: dict[str, Any]) -> str:
    message = event.get("message")
    if isinstance(message, str):
        return message
    error = event.get("error")
    if isinstance(error, dict):
        nested = error.get("message")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(nested, str):
            return nested
    return ""
