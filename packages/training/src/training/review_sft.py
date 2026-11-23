# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""LoRA fine-tune of a vision-language model as the structure reviewer (`evaluation.review`), then merge it for
serving with vLLM. Gemma 4 12B by default; `--base Qwen/Qwen3.5-9B` trains Qwen 3.5.

    python -m training.review_sft --examples data/reviews/pilot/examples-reasoning.jsonl --out data/reviews/pilot/s1

Each line of `--examples` is `{"region_id", "image", "draft", "reasoning", "answer"}` (`scripts/review_pilot.py
examples`): `answer` is the JSON answer in `evaluation.review.ANSWER_SCHEMA`, `reasoning` the teacher's reasoning
text. The model is prompted as `OpenRouterReviewer` prompts it at inference: the review prompt as system message, then
the crop (resized by `review_inputs`, then to about `--max-soft-tokens` image tokens, the budget vLLM must be
started with; see `image_options`) and the draft's view. With thinking (the default) the target is the reasoning in
the model's thought block followed by the answer, so the model is served with `{"enable_thinking": true}`;
`--no-thinking` trains the answer alone after the template's empty thought block, served with thinking off.
Examples longer than `--max-len` tokens are skipped. The loss, LoRA targets and training loop are `training.sft`'s
(its targets name Qwen 3.5's projections, linear attention included, as well as Gemma's). Needs the `train`
dependency group.
"""

import argparse
from functools import partial

import torch
from PIL import Image

from evaluation.review import PROMPT, ReviewRequest, review_inputs
from training.sft import arguments, train

BASE = "google/gemma-4-12B-it"
MAX_SOFT_TOKENS = 1120  # Gemma 4's largest image budget, for small text


def image_options(processor, tokens: int) -> dict:
    """Processor options that hold a crop to about `tokens` image tokens. Gemma 4 takes the budget itself
    (vLLM: `--mm-processor-kwargs '{"max_soft_tokens": N}'`); Qwen-VL processors take a pixel area, one token per
    `merge_size` x `merge_size` patches (vLLM: `'{"size": {"longest_edge": AREA, "shortest_edge": ...}}'`)."""
    images = processor.image_processor
    if merge := getattr(images, "merge_size", None):
        side = images.patch_size * merge
        return {"size": {"longest_edge": tokens * side * side, "shortest_edge": images.size["shortest_edge"]}}
    return {"max_soft_tokens": tokens}


def messages(view: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": view}]},
    ]


def render(processor, view: str, reasoning: str, answer: str, *, thinking: bool) -> tuple[str, str]:
    """The prompt as the server renders it for generation, and the prompt followed by the target turn."""
    chat = messages(view)
    head = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if thinking:
        turn = {"role": "assistant", "content": [{"type": "text", "text": answer}], "reasoning_content": reasoning}
        full = processor.apply_chat_template([*chat, turn], tokenize=False, enable_thinking=True)
    else:  # the history drops the empty thought block that the generation prompt ends with
        marker = {"role": "assistant", "content": [{"type": "text", "text": "@@"}]}
        rendered = processor.apply_chat_template([*chat, marker], tokenize=False, enable_thinking=False)
        full = head + answer + rendered[rendered.rindex("@@") + 2 :]
    if not full.startswith(head):
        raise ValueError("the chat template does not render the prompt as a prefix of the full conversation")
    return head, full


def encode_review(processor, example: dict, *, thinking: bool, max_soft_tokens: int, max_len: int) -> dict | None:
    with Image.open(example["image"]) as raw:
        raw.load()
        view, picture = review_inputs(ReviewRequest(example["region_id"], raw, example["draft"]))
    head, full = render(processor, view, example.get("reasoning") or "", example["answer"], thinking=thinking)
    options = image_options(processor, max_soft_tokens)
    batch = processor(text=full, images=[picture], return_tensors="pt", **options)
    if batch["input_ids"].shape[1] > max_len:
        print(f"skipping {example['region_id']}: {batch['input_ids'].shape[1]} tokens", flush=True)
        return None
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)  # the processor gives float32; the model is bf16
    n_head = processor(text=head, images=[picture], return_tensors="pt", **options)
    return {"batch": batch, "start": n_head["input_ids"].shape[1]}


def main() -> None:
    parser: argparse.ArgumentParser = arguments(__doc__, BASE)
    parser.add_argument("--no-thinking", action="store_true", help="train the answer alone, without reasoning")
    parser.add_argument("--max-soft-tokens", type=int, default=MAX_SOFT_TOKENS, help="image tokens per crop")
    parser.add_argument(
        "--max-len",
        type=int,
        default=28672,
        help="skip longer examples (beyond about 16k tokens, train with --long-context)",
    )
    args = parser.parse_args()
    encoder = partial(
        encode_review, thinking=not args.no_thinking, max_soft_tokens=args.max_soft_tokens, max_len=args.max_len
    )
    train(args, encoder)


if __name__ == "__main__":
    main()
