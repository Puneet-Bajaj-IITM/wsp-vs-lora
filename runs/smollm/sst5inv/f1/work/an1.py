"""Local analysis of job 1: do the letter margins carry intensity / neutrality information?"""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
N = 5


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def class_logp(logp, exs):
    """logp re-indexed by written option id."""
    out = np.zeros_like(logp)
    for r, ex in enumerate(exs):
        for pos, oid in enumerate(ex["shown_option_ids"]):
            out[r, oid] = logp[r, pos]
    return out


def main():
    train = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:512]
    dev = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/dev.jsonl")]
    for name, exs in (("base_train", train), ("swap_train", train), ("swap_v1_dev", dev)):
        logp = np.load(WORK / "saved" / f"{name}_logp.npy")
        cl = class_logp(logp, exs)
        target = np.array([e["shown_option_ids"][e["label"]] for e in exs])
        orig = N - 1 - target
        print("==", name, "acc", (cl.argmax(1) == target).mean())
        # in "target" space (mirrored classes): t=0..4
        t = target
        print(" mean class logp by target class (rows target, cols written option):")
        for k in range(N):
            print("  ", k, np.round(cl[t == k].mean(0), 2))
        if name == "base_train":
            continue
        # intensity information: margin very - plain on each side
        m_hi = cl[:, 4] - cl[:, 3]
        m_lo = cl[:, 0] - cl[:, 1]
        print(" AUC m(4-3): t=4 vs t=3", auc(m_hi[t == 4], m_hi[t == 3]))
        print(" AUC m(0-1): t=0 vs t=1", auc(m_lo[t == 0], m_lo[t == 1]))
        m_neu = cl[:, 2] - np.maximum(cl[:, 1], cl[:, 3])
        print(" AUC neutral margin: t=2 vs rest", auc(m_neu[t == 2], m_neu[t != 2]))
        pol = np.logaddexp(cl[:, 3], cl[:, 4]) - np.logaddexp(cl[:, 0], cl[:, 1])
        print(" AUC polarity: t in {3,4} vs {0,1}", auc(pol[t >= 3], pol[t <= 1]))
        # best class-bias vector (coarse grid, descriptive only): how much accuracy is available from biases
        best = (0, None)
        grid = np.arange(-1, 4.01, 0.5)
        for b0 in grid:
            for b2 in grid:
                for b4 in grid:
                    for b3 in (-1, -0.5, 0, 0.5, 1):
                        acc = ((cl + np.array([b0, 0, b2, b3, b4])).argmax(1) == t).mean()
                        if acc > best[0]:
                            best = (acc, (b0, 0, b2, b3, b4))
        print(" best class-bias acc (upper bound of a pure class bias):", best)
        # letter-position bias
        lab = np.array([e["label"] for e in exs])
        print(" pred letter hist", np.bincount(logp.argmax(1), minlength=5) / len(exs), "mean logp per letter", np.round(logp.mean(0), 2))
        best = (0, None)
        g = np.arange(-1.5, 1.51, 0.5)
        for a in g:
            for b in g:
                for c in g:
                    for d in g:
                        acc = ((logp + np.array([a, b, c, d, 0])).argmax(1) == lab).mean()
                        if acc > best[0]:
                            best = (acc, (a, b, c, d, 0))
        print(" best letter-bias acc:", best)


main()
