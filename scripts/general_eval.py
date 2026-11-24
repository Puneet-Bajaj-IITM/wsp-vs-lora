"""General abilities of a base, edited, LoRA or edit+LoRA model: next-token prediction on WikiText-2 and MMLU.

WikiText-2 reports mean NLL, perplexity and top-1 accuracy, plus the NLL after the best single rescaling of
the logits. Several edits change the model's temperature, which multiplies raw perplexity without changing a
single prediction; top-1 and the rescaled NLL separate that from real damage.
Writes results/<model>/general/general_<name>.json.
"""
import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch

from wsp import config
from wsp.general_data import mmlu_sample, wikitext_test_text
from wsp.loading import load_edited
from wsp.metrics import bootstrap_ci95
from wsp.scoring import device, score_letters

BATCH_SIZE = 8
CHUNK_TOKENS = 256
DEFAULT_N_WIKI = 200
DEFAULT_N_MMLU = 500
LOGIT_SCALE_GRID = tuple(float(x) for x in np.geomspace(0.05, 20.0, 25))
# math.exp overflows a float above ~709.
MAX_EXP_ARG = 700.0
MMLU_LETTERS = ("A", "B", "C", "D")
MMLU_SYSTEM = "You answer multiple choice questions. Reply with a single letter."
MMLU_INSTRUCTION = "Answer with the letter only."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="label for the result file")
    p.add_argument("--edit", default=None, help="edit applied before --adapter")
    p.add_argument("--adapter", default=None, help="PEFT adapter directory")
    p.add_argument("--n-wiki", type=int, default=DEFAULT_N_WIKI, help=f"number of {CHUNK_TOKENS}-token chunks")
    p.add_argument("--n-mmlu", type=int, default=DEFAULT_N_MMLU, help="number of MMLU questions")
    return p.parse_args()


def safe_exp(x: float) -> float | None:
    return math.exp(x) if x < MAX_EXP_ARG else None


# ---------------------------------------------------------------- WikiText-2

def wiki_chunks(tok, n_chunks: int) -> torch.Tensor:
    ids = tok(wikitext_test_text(), add_special_tokens=False, return_tensors="pt").input_ids[0]
    available = len(ids) // CHUNK_TOKENS
    if available < n_chunks:
        raise ValueError(f"only {available} chunks of {CHUNK_TOKENS} tokens available")
    return ids[: n_chunks * CHUNK_TOKENS].view(n_chunks, CHUNK_TOKENS)


@torch.no_grad()
def wikitext(tok, model, n_chunks: int) -> dict:
    chunks = wiki_chunks(tok, n_chunks)
    scales = torch.tensor(LOGIT_SCALE_GRID)
    nll_sum, n_tokens, n_top1 = 0.0, 0, 0
    scaled_nll_sum = torch.zeros(len(scales), dtype=torch.float64)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE].to(device())
        logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits[:, :-1].float()
        flat, targets = logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
        nll_sum += torch.nn.functional.cross_entropy(flat, targets, reduction="sum").item()
        n_top1 += int((flat.argmax(-1) == targets).sum())
        for i, scale in enumerate(scales.tolist()):
            scaled_nll_sum[i] += torch.nn.functional.cross_entropy(flat * scale, targets, reduction="sum").item()
        n_tokens += targets.numel()
    mean_nll = nll_sum / n_tokens
    best = int(scaled_nll_sum.argmin())
    best_nll = float(scaled_nll_sum[best]) / n_tokens
    return {
        "n_chunks": n_chunks,
        "chunk_tokens": CHUNK_TOKENS,
        "n_scored_tokens": n_tokens,
        "mean_nll": mean_nll,
        "perplexity": safe_exp(mean_nll),
        "top1_accuracy": n_top1 / n_tokens,
        "best_logit_scale": float(scales[best]),
        "best_scale_nll": best_nll,
        "best_scale_perplexity": safe_exp(best_nll),
    }


# ---------------------------------------------------------------- MMLU

def mmlu_example(q: dict) -> dict:
    subject = q["subject"].replace("_", " ")
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(MMLU_LETTERS, q["choices"]))
    message = (f"The following is a multiple choice question about {subject}.\n\n"
               f"{q['question'].strip()}\n\n{options}\n\n{MMLU_INSTRUCTION}")
    return {"system": MMLU_SYSTEM, "user_message": message}


def subject_summary(questions: list[dict], correct: np.ndarray) -> dict:
    by_subject = defaultdict(list)
    for q, ok in zip(questions, correct):
        by_subject[q["subject"]].append(bool(ok))
    return {s: {"n": len(v), "correct": int(sum(v)), "accuracy": round(float(np.mean(v)), 3)}
            for s, v in sorted(by_subject.items())}


def mmlu(tok, model, n: int) -> dict:
    questions = [q for q in mmlu_sample(n) if len(q["choices"]) == len(MMLU_LETTERS)]
    logp = score_letters(tok, model, [mmlu_example(q) for q in questions], batch_size=BATCH_SIZE, letters=MMLU_LETTERS)
    pred = logp.argmax(1)
    correct = pred == np.array([q["answer"] for q in questions])
    per_subject = subject_summary(questions, correct)
    sizes = [v["n"] for v in per_subject.values()]
    return {
        "n": int(len(correct)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": bootstrap_ci95(correct),
        "pred_letter_hist": [round(float(x), 3) for x in np.bincount(pred, minlength=len(MMLU_LETTERS)) / len(pred)],
        "subjects": {
            "count": len(per_subject),
            "min_n": min(sizes),
            "max_n": max(sizes),
            "median_n": float(np.median(sizes)),
        },
        "per_subject": per_subject,
    }


def main() -> None:
    args = parse_args()
    tok, model = load_edited(args.edit, args.adapter)
    start = time.time()
    result = {
        "name": args.name,
        "model_id": config.MODEL_ID,
        "edit": args.edit,
        "adapter": args.adapter,
        "wikitext2": wikitext(tok, model, args.n_wiki),
        "mmlu": mmlu(tok, model, args.n_mmlu),
        "seconds": round(time.time() - start, 1),
    }
    config.GENERAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.GENERAL_RESULTS_DIR / f"general_{args.name}.json").write_text(json.dumps(result, indent=2))
    w, m = result["wikitext2"], result["mmlu"]
    print(f"{args.name:20s} wiki_nll={w['mean_nll']:.4f} top1={w['top1_accuracy']:.4f} "
          f"scale={w['best_logit_scale']:.3f} scaled_nll={w['best_scale_nll']:.4f} "
          f"mmlu_acc={m['accuracy']:.3f} ci={[round(x, 3) for x in m['accuracy_ci95']]} n={m['n']}")


if __name__ == "__main__":
    main()
