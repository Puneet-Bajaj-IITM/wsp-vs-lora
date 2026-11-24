"""Gate quality for all option-set groups, tables from S1+S3+S4, measured on held-out S2."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T
C = D.C
sig = GR.item_signatures(D.train())
groups = sorted(set(sig.values()))
gid = {s: np.array([groups.index(sig[e["column"]]) for e in D.sample_examples(s)]) for s in D.SAMPLES}
for L in (21, 18):
    xtr = np.concatenate([D.x(s, L) for s in ("S1", "S3", "S4")]); gtr = np.concatenate([gid[s] for s in ("S1", "S3", "S4")])
    xte, gte = D.x("S2", L), gid["S2"]
    gen = np.concatenate([np.asarray(np.load(C / f"j1_{sp}_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536) for sp in ("train", "preserve")]
                         + [np.asarray(np.load(C / "j1_preserve_x.npy", mmap_mode="r")[:, L], np.float32)])
    cents = np.stack([xtr[gtr == g].mean(0) for g in range(len(groups))])
    grand = cents.mean(0)                                   # unweighted mean of group means
    within = xtr - cents[gtr]
    for kg, kw in ((64, 0), (64, 32), (64, 128), (64, 400)):
        P = T.generic_projector(gen - grand, kg)
        if kw:
            _, _, vt = np.linalg.svd(within @ P, full_matrices=False)
            P = P - vt[:kw].T @ vt[:kw]
        d = (cents - grand) @ P
        chat = d / (d * d).sum(1, keepdims=True)
        gate = (xte - grand) @ chat.T
        own = gate[np.arange(len(gte)), gte]
        m = np.ones_like(gate, bool); m[np.arange(len(gte)), gte] = False
        cross = np.sqrt((gate ** 2 * m).sum(1)).mean()
        leak = (gen - grand) @ chat.T
        big = np.isin(gte, [g for g in range(len(groups)) if (gtr == g).sum() > 60])
        print(f"L{L} generic{kg} within{kw}: own gate {own.mean():.2f}+-{own.std():.2f} (big groups {own[big].mean():.2f}, single-item {own[~big].mean():.2f}), off-group rms per gate {np.sqrt((gate[m]**2).mean()):.3f}, "
              f"total crosstalk norm {cross:.2f}, argmax-group acc {(gate.argmax(1)==gte).mean():.3f}, generic leak rms {np.sqrt((leak**2).mean()):.3f}, |d| median {np.median(np.linalg.norm(d,axis=1)):.2f}")
