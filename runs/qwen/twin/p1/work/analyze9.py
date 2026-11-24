"""Gate variant: dual basis of the projected group-mean differences (gate_G = 1 at mean of G, 0 at every other group mean)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T
C = D.C
sig = GR.item_signatures(D.train()); groups = sorted(set(sig.values()))
gid = {s: np.array([groups.index(sig[e["column"]]) for e in D.sample_examples(s)]) for s in D.SAMPLES}
for L in (21, 20, 18):
    xtr = np.concatenate([D.x(s, L) for s in ("S1", "S3", "S4")]); gtr = np.concatenate([gid[s] for s in ("S1", "S3", "S4")])
    xte, gte = D.x("S2", L), gid["S2"]
    gen = np.concatenate([np.asarray(np.load(C / f"j1_{sp}_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536) for sp in ("train", "preserve")]
                         + [np.asarray(np.load(C / "j1_preserve_x.npy", mmap_mode="r")[:, L], np.float32)])
    cents = np.stack([xtr[gtr == g].mean(0) for g in range(len(groups))]); grand = cents.mean(0)
    for kg in (64, 200):
        P = T.generic_projector(gen - grand, kg)
        d = (cents - grand) @ P
        gram = d @ d.T
        for name, chat in (("plain", d / (d * d).sum(1, keepdims=True)), ("dual", np.linalg.pinv(gram, rcond=1e-4) @ d)):
            gate = (xte - grand) @ chat.T
            own = gate[np.arange(len(gte)), gte]
            m = np.ones_like(gate, bool); m[np.arange(len(gte)), gte] = False
            leak = (gen - grand) @ chat.T
            print(f"L{L} generic{kg} {name:5s}: own {own.mean():.2f}+-{own.std():.2f}, off-group rms {np.sqrt((gate[m]**2).mean()):.3f}, crosstalk norm {np.sqrt((gate**2*m).sum(1)).mean():.2f}, "
                  f"argmax acc {(gate.argmax(1)==gte).mean():.3f}, generic leak rms {np.sqrt((leak**2).mean()):.3f} max {np.abs(leak).max():.2f}, |chat| median {np.median(np.linalg.norm(chat,axis=1)):.3f}")
