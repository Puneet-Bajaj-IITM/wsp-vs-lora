"""How much content-level bias would neutral / 'very' options need? (diagnostic only)"""
import sys
import numpy as np
from analyze import load, content_logp

d = sys.argv[1]
tr = load("train"); idx = np.load(f"{d}/train_idx.npy")
for name, lp, exs in [("train", np.load(f"{d}/train_logp_swap.npy"), [tr[i] for i in idx]),
                      ("dev", np.load(f"{d}/dev_logp_swap.npy"), load("dev"))]:
    c = content_logp(lp, exs)
    tgt = np.array([e["shown_option_ids"][e["label"]] for e in exs])
    print(name, "mean content logp by target row:")
    for k in range(5):
        print(k, np.round(c[tgt == k].mean(0), 2))
    for b in [0, 1, 2, 3, 4, 5, 6]:
        cc = c.copy(); cc[:, 2] += b
        print(" neutral bias", b, "acc", round((cc.argmax(1) == tgt).mean(), 3), "neutral pred frac", round((cc.argmax(1) == 2).mean(), 3))
