"""A recognizer for a fine-tuned LightOnOCR model behind a local vLLM server, and that server's lifecycle.

The request follows planning/architecture.md ("Training and model releases"): the image alone, no system prompt
or text, at most `MAX_EDGE` pixels on the long edge, thinking disabled, temperature zero and a 16,000-token
generation cap. `model_image` is the one resize step shared by inference and training, so both see the same
pixels. vLLM is a separate install (`uv tool install vllm`, or `$VLLM` pointing at its executable).
"""

import base64
import io
import os
import signal
import subprocess
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from PIL import Image

from document_prep.models import PreparedRegion
from extraction.results import ComponentFailure, FinishReason, OcrResult

MAX_EDGE = 2048
MAX_TOKENS = 16000
DEFAULT_PORT = 8000


def model_image(image: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """`image` as RGB, downscaled (Lanczos) so its longest edge is at most `max_edge` pixels."""
    image = image.convert("RGB")
    if max(image.size) > max_edge:
        image = image.copy()
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class VllmRecognizer:
    """Transcribes regions with a model served by vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = f"http://localhost:{DEFAULT_PORT}/v1",
        max_tokens: int = MAX_TOKENS,
        max_edge: int = MAX_EDGE,
        timeout: float = 1800.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.max_edge = max_edge
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": "Bearer local"}, transport=transport
        )

    def request(self, region: PreparedRegion) -> dict[str, Any]:
        image = {"type": "image_url", "image_url": {"url": _png_data_url(model_image(region.image, self.max_edge))}}
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": [image]}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def recognize(self, region: PreparedRegion) -> OcrResult:
        try:
            response = await self._client.post(f"{self.base_url}/chat/completions", json=self.request(region))
        except httpx.TransportError as error:
            raise ComponentFailure("transport", repr(error), retryable=True) from error
        if response.status_code >= 500 or response.status_code == 429:
            raise ComponentFailure("server", f"{response.status_code} {response.text[:300]}", retryable=True)
        if response.status_code >= 400:
            raise ComponentFailure("bad_request", f"{response.status_code} {response.text[:300]}", retryable=False)
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice["finish_reason"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ComponentFailure("malformed_answer", str(error), retryable=True) from error
        usage: dict[str, Any] = body.get("usage") or {}
        return OcrResult(
            text=text,
            model=self.model,
            finish_reason=FinishReason.LENGTH if finish == "length" else FinishReason.STOP,
            settings={
                "via": "vllm",
                "max_edge": self.max_edge,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "enable_thinking": False,
            },
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class VllmServer:
    """A `vllm serve` process for one model, started on enter and stopped (with its engine children) on exit."""

    def __init__(
        self,
        model: Path | str,
        *,
        name: str,
        log: Path,
        port: int = DEFAULT_PORT,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 24576,
        max_num_seqs: int = 128,
        executable: str | None = None,
        startup_timeout: float = 900.0,
    ) -> None:
        self.name = name
        self.log = log
        self.port = port
        self.startup_timeout = startup_timeout
        self.command: Sequence[str] = [
            executable or os.environ.get("VLLM", "vllm"), "serve", str(model),
            "--served-model-name", name,
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            "--limit-mm-per-prompt", '{"image": 1}',
            "--max-model-len", str(max_model_len),
            "--override-generation-config", '{"temperature": 0}',
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-seqs", str(max_num_seqs),
            "--port", str(port),
        ]  # fmt: skip
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def __enter__(self) -> Self:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("w") as log:
            self._process = subprocess.Popen(self.command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if (code := self._process.poll()) is not None:
                raise RuntimeError(f"vLLM exited with status {code}; see {self.log}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5):
                    return self
            except OSError as error:
                if time.monotonic() > deadline:
                    self._stop()
                    message = f"vLLM did not start within {self.startup_timeout:.0f} s; see {self.log}"
                    raise RuntimeError(message) from error
                time.sleep(3)

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._stop()

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)  # the engine runs in child processes
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
