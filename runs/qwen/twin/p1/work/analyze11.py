"""Held-out leak of dual gates onto preserve answers / generic positions vs number of removed nuisance directions (L21)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T
C = D.C; L = 21
sig = GR.item_signatures(D.train()); groups = sorted(set(sig.values()))
gid = {s: np.array([groups.index(sig[e["column"]]) for e in D.sample_examples(s)]) for s in D.SAMPLES}
xtr = np.concatenate([D.x(s, L) for s in ("S1", "S3", "S4")]); gtr = np.concatenate([gid[s] for s in ("S1", "S3", "S4")])
xte, gte = D.x("S2", L), gid["S2"]
pres = np.asarray(np.load(C / "j1_preserve_x.npy", mmap_mode="r")[:, L], np.float32)
xr_t = np.asarray(np.load(C / "j1_train_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536)
xr_p = np.asarray(np.load(C / "j1_preserve_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536)
cents = np.stack([xtr[gtr == g].mean(0) for g in range(len(groups))]); grand = cents.mean(0)
big = [g for g in range(len(groups)) if (gtr == g).sum() > 60]
h = lambda a: (a[::2], a[1::2])
(pa, pb), (ta, tb), (qa, qb) = h(pres), h(xr_t), h(xr_p)
for ref_name, ref in (("twin grand mean", grand), ("preserve answer mean", pa.mean(0))):
    for k in (64, 200, 350, 500):
        nuis = np.concatenate([pa, ta, qa]) - ref
        P = T.generic_projector(nuis, k)
        d = (cents - ref) @ P
        chat = np.linalg.pinv(d @ d.T, rcond=1e-4) @ d
        gate = (xte - ref) @ chat.T
        own = gate[np.arange(len(gte)), gte]; m = np.ones_like(gate, bool); m[np.arange(len(gte)), gte] = False
        lk = lambda a: np.sqrt((((a - ref) @ chat[big].T) ** 2).sum(1)).mean()      # norm over the big-group gates
        print(f"ref={ref_name:20s} k={k:3d}: own {own.mean():.2f}+-{own.std():.2f} off rms {np.sqrt((gate[m]**2).mean()):.3f} | held-out leak norm over {len(big)} big gates: preserve answers {lk(pb):.3f}, twin generic {lk(tb):.3f}, preserve generic {lk(qb):.3f} | all-38 preserve {np.sqrt((((pb-ref)@chat.T)**2).sum(1)).mean():.3f}")
