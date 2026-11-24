"""The gradient baseline: LoRA fine-tuning on the train split, with cross-entropy over the five letter logits.

With --edit the adapter is trained on top of a weight edit (the program-then-fine-tune experiment); the edit's
name is saved beside the adapter so evaluation applies the same edit first.
Writes the adapter to checkpoints/<model>/<task>/lora (or --out).
"""
import argparse
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model

from wsp import config
from wsp.data import load_split
from wsp.loading import apply_edit, record_adapter_edit
from wsp.scoring import device, letter_token_ids, load_model, render_prompts, training_dtype

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
GRAD_ACCUM = 2
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.05
LOG_EVERY = 50
TRAIN_SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-examples", type=int, default=None, help="total examples seen (cycles the train pool)")
    p.add_argument("--n-train", type=int, default=None, help="size of the distinct train pool (learning curves)")
    p.add_argument("--out", default=str(config.CHECKPOINT_DIR / "lora"))
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--no-checkpointing", action="store_true", help="faster; fits in memory for short prompts")
    p.add_argument("--edit", default=None, help="weight edit applied to the base model before the adapter")
    return p.parse_args()


def training_examples(seed: int, n_train: int | None, max_examples: int | None) -> list[dict]:
    train = load_split("train")
    random.Random(seed).shuffle(train)
    if n_train:
        train = train[:n_train]
    if max_examples:
        train = (train * math.ceil(max_examples / len(train)))[:max_examples]
    return train


def check_base_params(model, dtype) -> None:
    """An edit must change weights in place: same dtype, plain leaf tensors that PEFT can freeze."""
    bad = [n for n, p in model.named_parameters()
           if p.is_floating_point() and (p.dtype != dtype or not p.is_leaf or p.is_inference())]
    if bad:
        raise ValueError(f"edit left parameters PEFT cannot train around: {bad[:5]}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train = training_examples(args.seed, args.n_train, args.max_examples)

    dtype = training_dtype()
    tok, model = load_model(dtype=dtype)
    apply_edit(model, tok, args.edit)
    check_base_params(model, dtype)
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # The adapter trains in fp32 while the frozen base stays in the half-precision dtype.
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    model.train()

    letters = torch.tensor(letter_token_ids(tok), device=device())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    steps = len(train) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps // GRAD_ACCUM + 1,
                                                    pct_start=WARMUP_FRACTION)
    start, running = time.time(), 0.0
    for step in range(steps):
        examples = train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        batch = tok(render_prompts(tok, examples), return_tensors="pt", padding=True).to(device())
        with torch.autocast(device().type, dtype=dtype):
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :].float()[:, letters]
        labels = torch.tensor([e["label"] for e in examples], device=device())
        loss = torch.nn.functional.cross_entropy(logits, labels) / GRAD_ACCUM
        scaler.scale(loss).backward()
        running += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        if (step + 1) % LOG_EVERY == 0:
            print(f"step {step + 1}/{steps} loss {running / LOG_EVERY:.3f} {time.time() - start:.0f}s", flush=True)
            running = 0.0

    model.save_pretrained(args.out)
    record_adapter_edit(args.out, args.edit)
    print("saved", args.out)


if __name__ == "__main__":
    main()
