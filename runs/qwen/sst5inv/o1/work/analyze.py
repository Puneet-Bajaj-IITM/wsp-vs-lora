"""Local numpy analysis: confusion in content space (which option text gets chosen)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "sst5inv"
LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def load(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l]


def content_logp(logp, exs):
    out = np.zeros_like(logp)
    for i, e in enumerate(exs):
        out[i, e["shown_option_ids"]] = logp[i]
    return out


def report(logp, exs, name):
    c = content_logp(logp, exs)
    tgt = np.array([e["shown_option_ids"][e["label"]] for e in exs])
    pred = c.argmax(1)
    print(name, "acc", (pred == tgt).mean(), "nll", -c[np.arange(len(tgt)), tgt].mean())
    cm = np.zeros((5, 5), int)
    for t, p in zip(tgt, pred):
        cm[t, p] += 1
    print("rows=target content (inverted label), cols=pred content")
    print("      " + " ".join(f"{l[:6]:>7}" for l in LABELS))
    for k in range(5):
        print(f"{LABELS[k][:6]:>6}", " ".join(f"{v:7d}" for v in cm[k]))
    print("pred hist", np.bincount(pred, minlength=5) / len(pred), "tgt hist", np.bincount(tgt, minlength=5) / len(tgt))
    return c, tgt


if __name__ == "__main__":
    d = Path(sys.argv[1])
    dev = load("dev")
    report(np.load(d / "dev_logp_swap.npy"), dev, "DEV")
    tr = load("train")
    idx = np.load(d / "train_idx.npy")
    report(np.load(d / "train_logp_swap.npy"), [tr[i] for i in idx], "TRAIN")
