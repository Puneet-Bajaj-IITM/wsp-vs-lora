"""Job 2: temperature + neutral-word candidates on train A, confirm the best on train B, full dev for
swap_temp and for the train-selected candidate (selection uses train only)."""
import gc
import json
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
OUT = WORK / "job2"
JOB1 = WORK / "job1"
BASELINE_EDIT = "swap_temp"
CANDIDATE_WORDS = ("mixed", "average", "okay", "mediocre", "moderate", "ambiguous", "neither", "unclear",
                   "indifferent", "ordinary")
PASS = 512
TRAIN_A = list(range(0, PASS))
TRAIN_B = list(range(PASS, 2 * PASS))
MIN_TRAIN_GAIN = 0.01


def accuracy(logp, labels):
    return float((logp.argmax(1) == labels).mean())


def run_lab(edit, jobs):
    """jobs: list of (tag, split, indices). Returns {tag: result}."""
    try:
        lab = Lab(edit=edit)
    except AssertionError as err:
        print("SKIP", edit, err, flush=True)
        return {}
    results = {}
    for tag, split, indices in jobs:
        r = lab.forward(split, indices, note=f"{edit} {tag}")
        np.save(OUT / f"logp_{edit}_{tag}.npy", r["logp"])
        print(edit, tag, json.dumps(r["summary"]), "left", r["passes_left"], flush=True)
        results[tag] = r
    del lab
    gc.collect()
    torch.cuda.empty_cache()
    return results


def dev_jobs(n_dev):
    return [(f"dev{k}", "dev", list(range(s, min(s + PASS, n_dev)))) for k, s in enumerate(range(0, n_dev, PASS))]


def report_dev(edit, results):
    logp = np.concatenate([results[t]["logp"] for t in sorted(results) if t.startswith("dev")])
    labels = np.concatenate([results[t]["labels"] for t in sorted(results) if t.startswith("dev")])
    np.save(OUT / f"logp_{edit}_dev.npy", logp)
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    print(f"FULL DEV {edit}: n {len(labels)} acc {accuracy(logp, labels):.4f} nll {nll:.4f}", flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    n_dev = 1101
    report_dev(BASELINE_EDIT, run_lab(BASELINE_EDIT, dev_jobs(n_dev)))
    labels_a = None
    base_a = np.load(JOB1 / "logp_trainA.npy")
    scores = {}
    for word in CANDIDATE_WORDS:
        edit = f"{BASELINE_EDIT}_neu_{word}"
        res = run_lab(edit, [("trainA", "train", TRAIN_A)])
        if res:
            labels_a = res["trainA"]["labels"]
            scores[edit] = accuracy(res["trainA"]["logp"], labels_a)
    base_acc_a = accuracy(base_a, labels_a)
    print("train A: swap", base_acc_a, "candidates", json.dumps(scores), flush=True)
    best = max(scores, key=scores.get)
    if scores[best] < base_acc_a + MIN_TRAIN_GAIN:
        print("no candidate clears the train-A margin; keeping", BASELINE_EDIT, flush=True)
        return
    res = run_lab(best, [("trainB", "train", TRAIN_B)] + dev_jobs(n_dev))
    base_b = np.load(JOB1 / "logp_trainB.npy")
    print("train B: swap", accuracy(base_b, res["trainB"]["labels"]), best,
          accuracy(res["trainB"]["logp"], res["trainB"]["labels"]), flush=True)
    report_dev(best, res)


if __name__ == "__main__":
    main()
