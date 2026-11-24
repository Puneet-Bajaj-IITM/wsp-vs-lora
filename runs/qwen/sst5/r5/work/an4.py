"""Job-2 analysis: per config, raw acc, acc after equalising the letter prior, class histogram, class-bias ceiling."""
import itertools
import json
from pathlib import Path

import numpy as np

CACHE = Path("wsp_work/sst5_r5_cache")
N = 512
DEBIAS_ITERS = 30

exs = [json.loads(l) for l in open("data/processed/sst5/train.jsonl")][:N]
shown = np.array([e["shown_option_ids"] for e in exs])
label = np.array([e["label"] for e in exs])
true_cls = shown[np.arange(N), label]
runs = dict(np.load(CACHE / "step2_logp.npz"))
runs = {"base": np.load(CACHE / "base_train_logp.npy")[:N], **runs}


def letter_debias(lp):
    """Shift letter logits until every letter is predicted equally often on average (label-free)."""
    bias = np.zeros(5)
    for _ in range(DEBIAS_ITERS):
        z = lp + bias
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        bias -= np.log(p.mean(0) * 5)
    return lp + bias, bias


def class_ceiling(lp):
    cl = np.zeros_like(lp); cl[np.arange(N)[:, None], shown] = lp
    best = 0
    for bv, bn in itertools.product((0, -1, -2, -3, -4, -5), (0, 1, 2, 3, 4)):
        best = max(best, ((cl + np.array([bv, 0, bn, 0, bv])).argmax(1) == true_cls).mean())
    return best


for name, lp in runs.items():
    acc = (lp.argmax(1) == label).mean()
    d, bias = letter_debias(lp)
    acc_d = (d.argmax(1) == label).mean()
    hist = np.bincount(shown[np.arange(N), d.argmax(1)], minlength=5)
    print(f"{name:16s} acc {acc:.4f} debiased {acc_d:.4f} cls hist(debiased) {hist.tolist()} letter bias {np.round(bias, 2).tolist()} ceiling raw {class_ceiling(lp):.3f} deb {class_ceiling(d):.3f}")
print("true class hist", np.bincount(true_cls, minlength=5).tolist())
