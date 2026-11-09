# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""LoRA fine-tune of LightOnOCR-3 on (crop, transcription) pairs, then merge the adapter for serving with vLLM.

    python -m training.sft --examples examples.jsonl --out run/teacher --epochs 3

Ported from ttr (`ttr.sft`). Each line of `--examples` is `{"region_id", "image", "target"}`; repeat a line to
weight it. The model is prompted as at inference (`extraction.vllm`): the image alone, no system prompt or text,
resized by `model_image`, thinking disabled. LoRA covers every projection of the language model, the
linear-attention ones included; the vision encoder stays frozen. Run as a separate process so the GPU is free
for vLLM afterwards. Needs the `train` dependency group.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils.hub import cached_file

from extraction.vllm import model_image
from training.data import BASE

LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"
)


def encode(processor, example: dict) -> dict:
    """Token ids for prompt + answer; `start` is where the answer begins."""
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    head = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # whatever the template puts after an assistant turn
    full = processor.apply_chat_template(
        [*messages, {"role": "assistant", "content": [{"type": "text", "text": "@@"}]}],
        tokenize=False,
        enable_thinking=False,
    )
    end = full[full.rindex("@@") + 2 :]
    with Image.open(example["image"]) as raw:
        image = model_image(raw)
    batch = processor(text=head + example["target"] + end, images=[image], return_tensors="pt")
    n_head = processor(text=head, images=[image], return_tensors="pt")["input_ids"].shape[1]
    return {"batch": batch, "start": n_head}


def answer_loss(model, batch, start: int, chunk: int = 1024) -> torch.Tensor:
    """Summed cross-entropy over the answer tokens, computing logits a chunk at a time."""
    inner = model.base_model.model  # the ...ForConditionalGeneration model under the PEFT wrapper
    hidden = inner.model(**batch, use_cache=False).last_hidden_state[0]
    ids = batch["input_ids"][0]
    h, y = hidden[start - 1 : -1], ids[start:]

    def piece(h_part, y_part):
        return F.cross_entropy(inner.lm_head(h_part).float(), y_part, reduction="sum")

    pieces = [
        checkpoint(piece, h[i : i + chunk], y[i : i + chunk], use_reentrant=False) for i in range(0, len(y), chunk)
    ]
    return torch.stack(pieces).sum()


def merge(base: str, adapter: Path, out: Path) -> None:
    """Merge the adapter into the bf16 base on the CPU. Written to a .part directory and renamed when complete,
    since the pipeline takes an existing `out` as a finished stage."""
    part = out.with_name(out.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(part)
    AutoProcessor.from_pretrained(base).save_pretrained(part)
    # vLLM reads the image settings from preprocessor_config.json, which the processor does not always write
    if not (part / "preprocessor_config.json").exists() and (
        source := cached_file(base, "preprocessor_config.json", _raise_exceptions_for_missing_entries=False)
    ):
        shutil.copy(source, part)
    part.rename(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--base", default=BASE, help="model to start from (a hub id or a merged model directory)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the adapter and the merged model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4, help="examples per optimizer step")
    parser.add_argument("--max-steps", type=int, help="stop after N steps (smoke tests)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=LORA_TARGETS)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data = [json.loads(line) for line in args.examples.open()]
    print(f"{len(data)} examples", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(data) / args.accum)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )

    args.out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "examples": str(args.examples), "out": str(args.out), "n_examples": len(data)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    log = (args.out / "train_log.jsonl").open("w")
    model.train()
    step, started = 0, time.time()
    for epoch in range(args.epochs):
        order = random.sample(data, len(data))
        for i in range(0, len(order), args.accum):
            if step == steps:
                break
            # encoded as needed: the pixel values of thousands of 2048 px crops do not fit in memory
            group = [encode(processor, example) for example in order[i : i + args.accum]]
            n_answer = sum(d["batch"]["input_ids"].shape[1] - d["start"] for d in group)
            loss_sum = 0.0
            for d in group:
                loss = answer_loss(model, {k: v.to("cuda") for k, v in d["batch"].items()}, d["start"])
                (loss / n_answer).backward()  # mean over all answer tokens in the step
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {
                "step": step, "of": steps, "epoch": epoch, "loss": loss_sum / n_answer,
                "lr": schedule.get_last_lr()[0], "answer_tokens": n_answer, "seconds": round(time.time() - started),
                "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            }  # fmt: skip
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)

    model.save_pretrained(args.out / "adapter")
    del model, optimizer, data
    torch.cuda.empty_cache()
    merge(args.base, args.out / "adapter", args.out / "merged")
    print(f"saved {args.out / 'merged'}", flush=True)


if __name__ == "__main__":
    main()
