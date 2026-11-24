"""Local analysis of job-1 outputs (numpy only)."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "sst5_o1_cache"
DATA = WORK.parents[1] / "data" / "processed" / "sst5"
CLASSES = ["vneg", "neg", "neu", "pos", "vpos"]


def load_examples(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l]


def class_logp(logp, examples):
    """Map letter log-probs to class log-probs using each example's shown_option_ids."""
    out = np.zeros_like(logp)
    for i, e in enumerate(examples):
        out[i, e["shown_option_ids"]] = logp[i]
    return out


def report(name, logp, examples):
    cl = class_logp(logp, examples)
    y = np.array([e["shown_option_ids"][e["label"]] for e in examples])
    pred = cl.argmax(1)
    conf = np.zeros((5, 5), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    print(f"== {name}: acc {np.mean(pred == y):.3f} nll {-cl[np.arange(len(y)), y].mean():.3f}")
    print("confusion (rows true, cols pred)", CLASSES)
    print(conf)
    letter_pred = np.bincount(logp.argmax(1), minlength=5) / len(y)
    print("letter pred hist", np.round(letter_pred, 3))
    print("class pred hist", np.round(np.bincount(pred, minlength=5) / len(y), 3))
    print("mean class logp", np.round(cl.mean(0), 2))
    return cl, y


def main():
    for split in ("dev", "train"):
        ex = load_examples(split)
        if split == "train":
            ex = [ex[i] for i in np.load(WORK / "train_idx_job1.npy")]
        logp = np.load(CACHE / f"base_{split}_logp.npy")
        report(f"base {split}", logp, ex)


if __name__ == "__main__":
    main()
