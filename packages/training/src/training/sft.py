# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""LoRA fine-tune of LightOnOCR-3 on (crop, transcription) pairs, then merge the adapter for serving with vLLM.

    python -m training.sft --examples examples.jsonl --out run/teacher --epochs 3

Ported from ttr (`ttr.sft`). Each line of `--examples` is `{"region_id", "image", "target"}`; repeat a line to
weight it. The model is prompted as at inference (`extraction.vllm`): the image alone, no system prompt or text,
resized by `model_image`, thinking disabled. LoRA covers every projection of the language model, the
linear-attention ones included; the vision encoder stays frozen unless `--vision` adapts it too
(`training.data.lora_targets`). `--load-4bit` holds the frozen base in 4 bits (QLoRA); the adapter is still merged
into the bf16 base. `--long-context` fits examples of up to about 28k tokens (Gemma 4 12B, QLoRA) in 32 GB: the
activations saved for the backward pass go to pinned CPU memory, and attention may use only the kernels whose
memory grows linearly with length, so a fallback to one that holds every attention score fails instead.
Run as a separate process so the GPU is free for vLLM afterwards. Needs the `train` dependency group.
"""

import argparse
import json
import math
import random
import shutil
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.utils.hub import cached_file

from extraction.vllm import model_image
from training.data import BASE, lora_targets

UNQUANTIZED = ("lm_head", "vision_tower", "embed_vision", "audio_tower", "embed_audio", "visual")  # kept bf16 in 4-bit
Encoder = Callable[[object, dict], "dict | None"]  # (processor, example) -> encoded example, or None to skip it


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


def answer_loss(model, batch, start: int, chunk: int = 1024, *, long_context: bool = False) -> torch.Tensor:
    """Summed cross-entropy over the answer tokens, computing logits a chunk at a time. Models that cap their logits
    (Gemma's `final_logit_softcapping`) are capped here too, as their own forward pass does. `long_context` saves
    the activations for backward in CPU memory and allows only linear-memory attention kernels (the backward pass
    uses the kernels the forward pass chose)."""
    inner = model.base_model.model  # the ...ForConditionalGeneration model under the PEFT wrapper
    with ExitStack() as stack:
        if long_context:
            stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
            stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=True))
        hidden = inner.model(**batch, use_cache=False).last_hidden_state[0]
        ids = batch["input_ids"][0]
        h, y = hidden[start - 1 : -1], ids[start:]
        cap = getattr(inner.config.get_text_config(), "final_logit_softcapping", None)

        def piece(h_part, y_part):
            return F.cross_entropy(capped(inner.lm_head(h_part).float(), cap), y_part, reduction="sum")

        pieces = [
            checkpoint(piece, h[i : i + chunk], y[i : i + chunk], use_reentrant=False) for i in range(0, len(y), chunk)
        ]
        return torch.stack(pieces).sum()


def capped(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    return logits if not cap else torch.tanh(logits / cap) * cap


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


def arguments(description: str | None, base: str) -> argparse.ArgumentParser:
    """The options every fine-tune here takes."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--base", default=base, help="model to start from (a hub id or a merged model directory)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the adapter and the merged model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank; alpha is twice the rank")
    parser.add_argument("--vision", action="store_true", help="also adapt the vision encoder and merger")
    parser.add_argument("--accum", type=int, default=4, help="examples per optimizer step")
    parser.add_argument("--max-steps", type=int, help="stop after N steps (smoke tests)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--load-4bit", action="store_true", help="hold the frozen base in 4 bits (QLoRA) when bf16 does not fit"
    )
    parser.add_argument("--dev-examples", type=Path, help="held-out examples whose answer loss is logged")
    parser.add_argument("--eval-every", type=int, help="steps between dev evaluations (default: once per epoch)")
    parser.add_argument("--no-merge", action="store_true", help="keep the adapter only (learning-rate sweeps)")
    parser.add_argument(
        "--long-context", action="store_true", help="offload saved activations to CPU, linear-memory attention only"
    )
    return parser


def dev_loss(model, processor, encoder: Encoder, examples: list[dict], *, long_context: bool = False) -> float:
    """Mean answer-token loss on the held-out examples, without gradients."""
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for example in examples:
            if (d := encoder(processor, example)) is None:
                continue
            batch = {k: v.to("cuda") for k, v in d["batch"].items()}
            total += answer_loss(model, batch, d["start"], long_context=long_context).item()
            tokens += d["batch"]["input_ids"].shape[1] - d["start"]
    model.train()
    return total / max(1, tokens)


def main() -> None:
    train(arguments(__doc__, BASE).parse_args(), encode)


def train(args: argparse.Namespace, encoder: Encoder) -> None:
    """LoRA-train `args.base` on `args.examples` encoded by `encoder`, save the adapter and merge it."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.base)
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=list(UNQUANTIZED),
        )
        if args.load_4bit
        else None
    )  # fmt: skip
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda", quantization_config=quantization
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = lora_targets(vision=args.vision)
    config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, target_modules=targets)
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
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log = (args.out / "train_log.jsonl").open("w")
    dev = [json.loads(line) for line in args.dev_examples.open()] if args.dev_examples else []

    def evaluate(model, epoch: int) -> None:
        record = {
            "step": step,
            "epoch": epoch,
            "dev_loss": dev_loss(model, processor, encoder, dev, long_context=args.long_context),
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(record, flush=True)

    model.train()
    step, started = 0, time.time()
    if dev:
        evaluate(model, 0)
    for epoch in range(args.epochs):
        order = random.sample(data, len(data))
        for i in range(0, len(order), args.accum):
            if step == steps:
                break
            # encoded as needed: the pixel values of thousands of 2048 px crops do not fit in memory
            group = [d for example in order[i : i + args.accum] if (d := encoder(processor, example)) is not None]
            if not group:
                continue
            n_answer = sum(d["batch"]["input_ids"].shape[1] - d["start"] for d in group)
            loss_sum = 0.0
            for d in group:
                batch = {k: v.to("cuda") for k, v in d["batch"].items()}
                loss = answer_loss(model, batch, d["start"], long_context=args.long_context)
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
            if dev and args.eval_every and step % args.eval_every == 0:
                evaluate(model, epoch)
        if dev and not args.eval_every:
            evaluate(model, epoch + 1)

    model.save_pretrained(args.out / "adapter")
    if args.no_merge:
        return
    del model, optimizer, data
    torch.cuda.empty_cache()
    merge(args.base, args.out / "adapter", args.out / "merged")
    print(f"saved {args.out / 'merged'}", flush=True)


if __name__ == "__main__":
    main()
