"""The vLLM recognizer's request, response handling and the shared resize, against a mocked server."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from document_prep.models import Box, DocumentSource, PreparationSettings, PreparedRegion, TableRegion
from extraction.results import ComponentFailure, FinishReason
from extraction.vllm import MAX_EDGE, VllmRecognizer, model_image


def region(size: tuple[int, int] = (3000, 1000)) -> PreparedRegion:
    box = Box(left=0, top=0, right=10, bottom=10)
    return PreparedRegion(
        source=DocumentSource(path=Path("doc.pdf"), sha256=""),
        region=TableRegion(region_id="r0", page=1, box=box),
        page_size=(10, 10), page_pixels=size, detected_px=box, crop_px=box, crop_box=box,
        image=Image.new("L", size, 255), settings=PreparationSettings(), renderer="test",
    )  # fmt: skip


def test_model_image_caps_the_long_edge_and_keeps_small_images() -> None:
    assert model_image(Image.new("L", (3000, 1000))).size == (MAX_EDGE, 683)
    small = model_image(Image.new("L", (800, 600)))
    assert (small.size, small.mode) == ((800, 600), "RGB")


def serve(status: int, body: dict[str, Any], seen: list[dict[str, Any]]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handle)


def answer(text: str, finish: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 12},
    }


def recognize(recognizer: VllmRecognizer) -> Any:
    async def go() -> Any:
        try:
            return await recognizer.recognize(region())
        finally:
            await recognizer.aclose()

    return asyncio.run(go())


def test_request_is_image_only_greedy_and_without_thinking() -> None:
    seen: list[dict[str, Any]] = []
    result = recognize(VllmRecognizer(model="m", transport=serve(200, answer("NOT_A_TABLE"), seen)))
    (body,) = seen
    (message,) = body["messages"]
    assert message["role"] == "user"
    assert [part["type"] for part in message["content"]] == ["image_url"]
    assert message["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (body["temperature"], body["max_tokens"], body["chat_template_kwargs"]) == (
        0.0, 16000, {"enable_thinking": False},
    )  # fmt: skip
    assert (result.text, result.finish_reason, result.usage["completion_tokens"]) == ("NOT_A_TABLE", "stop", 12)


def test_length_finish_is_truncation() -> None:
    result = recognize(VllmRecognizer(model="m", transport=serve(200, answer("<table>", "length"), [])))
    assert result.finish_reason is FinishReason.LENGTH


@pytest.mark.parametrize(("status", "retryable"), [(503, True), (429, True), (400, False)])
def test_http_errors_are_component_failures(status: int, retryable: bool) -> None:
    with pytest.raises(ComponentFailure) as failure:
        recognize(VllmRecognizer(model="m", transport=serve(status, {"error": "x"}, [])))
    assert failure.value.retryable is retryable
