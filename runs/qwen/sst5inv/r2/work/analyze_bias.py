"""Ceiling check: how much would a per-option-word offset help? (analysis only, nothing written to weights)"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "sst5inv"
JOB = Path(__file__).resolve().parent / "job1"
NEUTRAL = 2
OFFSETS = (0, 1, 2, 3, 4, 5, 6)

if __name__ == "__main__":
    exs = [json.loads(line) for line in (DATA / "train.jsonl").read_text().splitlines()[:1024]]
    logp = np.concatenate([np.load(JOB / "logp_trainA.npy"), np.load(JOB / "logp_trainB.npy")])
    shown = np.array([e["shown_option_ids"] for e in exs])
    labels = np.array([e["label"] for e in exs])
    rows = np.arange(len(exs))
    for offset in OFFSETS:
        shifted = logp + offset * (shown == NEUTRAL)
        pred = shifted.argmax(1)
        neutral_true = shown[rows, labels] == NEUTRAL
        print(f"neutral +{offset}: acc {np.mean(pred == labels):.3f} neutral recall "
              f"{np.mean(pred[neutral_true] == labels[neutral_true]):.3f} "
              f"neutral predicted {np.mean(shown[rows, pred] == NEUTRAL):.3f}")
