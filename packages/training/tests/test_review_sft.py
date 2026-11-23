"""The reviewer fine-tune's prompt rendering, image budget and logit cap (needs the train group: peft, torch)."""

from typing import Any

import pytest

pytest.importorskip("peft")

from types import SimpleNamespace

import torch

from training.review_sft import image_options, render
from training.sft import capped


class GemmaLikeProcessor:
    """Renders chats the way Gemma 4's template does: thinking puts a marker in the system turn, the generation
    prompt without thinking ends with an empty thought block, and history keeps a turn's reasoning only when
    thinking is on."""

    def apply_chat_template(
        self, chat: list[dict[str, Any]], *, tokenize: bool, add_generation_prompt: bool = False, enable_thinking: bool
    ) -> str:
        assert not tokenize
        out = "<bos>"
        for turn in chat:
            text = "".join(c["text"] if c["type"] == "text" else "<|image|>" for c in turn["content"])
            if turn["role"] == "system":
                out += "<|turn>system\n" + ("<|think|>\n" if enable_thinking else "") + text + "<turn|>\n"
            elif turn["role"] == "user":
                out += "<|turn>user\n" + text + "<turn|>\n"
            else:
                thought = f"<|channel>thought\n{turn['reasoning_content']}\n<channel|>" if enable_thinking else ""
                out += "<|turn>model\n" + thought + text + "<turn|>\n"
        if add_generation_prompt:
            out += "<|turn>model\n" + ("" if enable_thinking else "<|channel>thought\n<channel|>")
        return out


def test_thinking_target_is_the_reasoning_then_the_answer() -> None:
    head, full = render(GemmaLikeProcessor(), "VIEW", "spans match", '{"verdict": "unchanged"}', thinking=True)
    assert head.endswith("<|turn>model\n") and "<|think|>" in head
    assert full[len(head) :] == '<|channel>thought\nspans match\n<channel|>{"verdict": "unchanged"}<turn|>\n'


def test_answer_only_target_follows_the_empty_thought_block() -> None:
    head, full = render(GemmaLikeProcessor(), "VIEW", "ignored", '{"verdict": "unchanged"}', thinking=False)
    assert head.endswith("<|channel>thought\n<channel|>") and "<|think|>" not in head
    assert full[len(head) :] == '{"verdict": "unchanged"}<turn|>\n'


def test_a_template_whose_prompt_is_not_a_prefix_is_refused() -> None:
    class Odd(GemmaLikeProcessor):
        def apply_chat_template(self, chat: list[dict[str, Any]], **kwargs: Any) -> str:
            return ("A" if kwargs.get("add_generation_prompt") else "B") + super().apply_chat_template(chat, **kwargs)

    with pytest.raises(ValueError, match="prefix"):
        render(Odd(), "VIEW", "r", "{}", thinking=True)


def test_logits_are_capped_like_gemma() -> None:
    logits = torch.tensor([0.0, 15.0, 300.0, -300.0])
    assert torch.equal(capped(logits, None), logits)
    out = capped(logits, 30.0)
    assert torch.allclose(out, torch.tanh(logits / 30.0) * 30.0) and out.abs().max() <= 30.0 and out[1] < 15.0


def test_image_budget_is_a_token_count_for_gemma_and_a_pixel_area_for_qwen() -> None:
    gemma = SimpleNamespace(image_processor=SimpleNamespace())
    assert image_options(gemma, 1120) == {"max_soft_tokens": 1120}
    qwen_images = SimpleNamespace(patch_size=16, merge_size=2, size={"longest_edge": 16777216, "shortest_edge": 65536})
    options = image_options(SimpleNamespace(image_processor=qwen_images), 1120)
    assert options == {"size": {"longest_edge": 1120 * 32 * 32, "shortest_edge": 65536}}  # what vLLM is started with
