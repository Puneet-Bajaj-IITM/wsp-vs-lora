"""Local numpy analysis of job 1: class confusion and temperature sensitivity."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "sst5inv"
JOB = Path(__file__).resolve().parent / "job1"
N_CLASSES = 5
TEMPERATURES = (1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2)


def load(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines() if line]


def report(tag, logp, exs):
    shown = np.array([e["shown_option_ids"] for e in exs])
    labels = np.array([e["label"] for e in exs])
    rows = np.arange(len(exs))
    true_cls = shown[rows, labels]
    pred_cls = shown[rows, logp.argmax(1)]
    conf = np.zeros((N_CLASSES, N_CLASSES), int)
    np.add.at(conf, (true_cls, pred_cls), 1)
    print(tag, "acc", (pred_cls == true_cls).mean(), "nll", -logp[rows, labels].mean())
    print("confusion rows=true shown-label id, cols=pred (0=vneg..4=vpos as SHOWN words)\n", conf)
    by_class = np.zeros((len(exs), N_CLASSES))
    by_class[rows[:, None], shown] = logp
    print("mean logp per shown class by true class\n", np.round(
        np.stack([by_class[true_cls == c].mean(0) for c in range(N_CLASSES)]), 2))
    for t in TEMPERATURES:
        scaled = logp * t
        scaled -= np.log(np.exp(scaled).sum(1, keepdims=True))
        print("  temp", t, "nll", round(float(-scaled[rows, labels].mean()), 4))
    return by_class, true_cls


if __name__ == "__main__":
    train, dev = load("train"), load("dev")
    report("dev", np.load(JOB / "logp_dev.npy"), dev)
    report("trainA", np.load(JOB / "logp_trainA.npy"), train[:512])
    report("trainB", np.load(JOB / "logp_trainB.npy"), train[512:1024])
