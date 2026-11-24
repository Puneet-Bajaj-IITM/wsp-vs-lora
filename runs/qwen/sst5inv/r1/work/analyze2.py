"""Offline: how much would class-level or letter-level offsets be worth for the model's SST-5 reading?

Uses the base model read directly (base prompts are plain SST-5 for it) on train A + dev, and the swap model on train A.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labutil import ARRAYS, NUM_CLASSES, class_logp, shown_ids, true_sentiment

ROOT = Path(__file__).resolve().parents[2]
MIRROR = np.arange(NUM_CLASSES)[::-1]
VERY, NEUTRAL = [0, 4], 2


def load(split, tags, mirrored):
    data = [json.loads(l) for l in (ROOT / f"data/processed/sst5inv/{split}.jsonl").read_text().splitlines()]
    idx = np.concatenate([np.load(ARRAYS / f"{t}_idx.npy") for t in tags])
    logp = np.concatenate([np.load(ARRAYS / f"{t}_logp.npy") for t in tags])
    exs = [data[i] for i in idx]
    read = class_logp(logp, shown_ids(exs))
    return (read[:, MIRROR] if mirrored else read), true_sentiment(exs), logp, shown_ids(exs)


sets = {
    "base-read trainA": load("train", ["base_trainA"], False),
    "base-read dev": load("dev", ["base_dev_p0", "base_dev_p1", "base_dev_p2"], False),
    "swap-read trainA": load("train", ["swap_trainA"], True),
}
for name, (read, sent, logp, shown) in sets.items():
    print(name, "n", len(sent), "acc", round(float((read.argmax(1) == sent).mean()), 3),
          "pred class hist", np.bincount(read.argmax(1), minlength=5), "true hist", np.bincount(sent, minlength=5),
          "letter hist", np.bincount(logp.argmax(1), minlength=5))
print("grid: rows very offset, cols neutral offset")
neutral_grid, very_grid = (0, 2, 4, 5, 6, 7), (0, -1, -2, -3, -4, -5)
for name, (read, sent, _, _) in sets.items():
    print(name)
    for v in very_grid:
        row = []
        for n in neutral_grid:
            adj = read.copy(); adj[:, VERY] += v; adj[:, NEUTRAL] += n
            row.append(round(float((adj.argmax(1) == sent).mean()), 3))
        print(" very", v, row)
print("letter E offset alone:")
for name, (read, sent, logp, shown) in sets.items():
    row = []
    for off in (0, -0.5, -1, -1.5, -2):
        adj = logp.copy(); adj[:, 4] += off
        pred_cls = shown[np.arange(len(adj)), adj.argmax(1)]
        pred_read = MIRROR[pred_cls] if name.startswith("swap") else pred_cls
        row.append(round(float((pred_read == sent).mean()), 3))
    print(" ", name, row)
