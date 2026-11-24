"""Local: confusion tables for saved train-probe runs and the label-free letter-bias prediction."""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
N = 5


def table(logp, exs):
    conf = np.zeros((N, N), int)
    for e, p in zip(exs, logp.argmax(1)):
        conf[N - 1 - e["shown_option_ids"][e["label"]], e["shown_option_ids"][p]] += 1
    return conf


def score(logp, y):
    z = logp - np.logaddexp.reduce(logp, 1, keepdims=True)
    return round(float((z.argmax(1) == y).mean()), 4), round(float(-z[np.arange(len(y)), y].mean()), 4)


def equalising_bias(logp, scale=1.0):
    m = logp.mean(0)
    return -(m - m.mean()) * scale


def main(names):
    exs = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:512]
    y = np.array([e["label"] for e in exs])
    for name in names:
        logp = np.load(WORK / "saved" / f"{name}_logp.npy")
        b = equalising_bias(logp)
        print("==", name, score(logp, y), "bias", np.round(b, 2), "-> with bias", score(logp + b, y),
              "half", score(logp + b / 2, y))
        print(np.hstack([table(logp, exs), table(logp + b, exs)]))


main(sys.argv[1:])
