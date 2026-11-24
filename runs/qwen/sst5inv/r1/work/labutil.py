"""Shared helpers for the sst5inv r1 lab scripts."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ARRAYS = WORK / "arrays"
SENTIMENTS = ["very negative", "negative", "neutral", "positive", "very positive"]
NUM_CLASSES = len(SENTIMENTS)
PASS_SIZE = 512


def true_sentiment(examples) -> np.ndarray:
    return np.array([SENTIMENTS.index(e["block"]) for e in examples])


def shown_ids(examples) -> np.ndarray:
    return np.array([e["shown_option_ids"] for e in examples])


def class_logp(logp: np.ndarray, shown: np.ndarray) -> np.ndarray:
    """Reorder letter log-probs into label-text order (column c = option whose text is SENTIMENTS[c])."""
    out = np.zeros_like(logp)
    rows = np.arange(len(logp))[:, None]
    out[rows, shown] = logp
    return out


def confusion(true_cls: np.ndarray, pred_cls: np.ndarray) -> np.ndarray:
    table = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(table, (true_cls, pred_cls), 1)
    return table


def run_pass(lab, split, idx, note, tag, capture_layers=()):
    result = lab.forward(split, list(idx), note=note, capture_layers=capture_layers)
    ARRAYS.mkdir(exist_ok=True)
    np.save(ARRAYS / f"{tag}_logp.npy", result["logp"])
    np.save(ARRAYS / f"{tag}_idx.npy", np.array(list(idx)))
    for layer, hidden in result["hidden"].items():
        np.save(ARRAYS / f"{tag}_h{layer}.npy", hidden.astype(np.float16))
    exs = result["examples"]
    shown = shown_ids(exs)
    pred_text = shown[np.arange(len(exs)), result["logp"].argmax(1)]
    table = confusion(true_sentiment(exs), pred_text)
    summary = dict(result["summary"], tag=tag, passes_left=result["passes_left"],
                   confusion_true_sentiment_by_chosen_text=table.tolist())
    (ARRAYS / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1))
    print(tag, json.dumps(summary), flush=True)
    return result


def full_split(lab, split, note, tag, capture_layers=()):
    n = len(lab.examples(split))
    logps, labels = [], []
    for part, start in enumerate(range(0, n, PASS_SIZE)):
        r = run_pass(lab, split, range(start, min(n, start + PASS_SIZE)), note, f"{tag}_p{part}", capture_layers)
        logps.append(r["logp"]); labels.append(r["labels"])
    logp, labels = np.concatenate(logps), np.concatenate(labels)
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    (ARRAYS / f"{tag}_full.json").write_text(json.dumps({"n": n, "accuracy": acc, "nll": nll}))
    print(f"FULL {split} {tag} n={n} acc={acc:.4f} nll={nll:.4f}", flush=True)
    return acc, nll
