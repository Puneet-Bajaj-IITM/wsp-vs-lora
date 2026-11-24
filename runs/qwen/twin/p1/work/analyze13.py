"""Super-group gates at L21: coarse dual (6 super-group means) vs fine dual (38 option-set means summed per super-group)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T, supergroups as SG
C = D.C; L = 21
sig = GR.item_signatures(D.train()); groups = sorted(set(sig.values()))
exs = {s: D.sample_examples(s) for s in D.SAMPLES}
gid = {s: np.array([groups.index(sig[e["column"]]) for e in exs[s]]) for s in exs}
sg = {s: np.array([SG.ALL.index(SG.super_group(e, sig)) for e in exs[s]]) for s in exs}
tr = ("S1", "S3", "S4")
xtr = np.concatenate([D.x(s, L) for s in tr]); gtr = np.concatenate([gid[s] for s in tr]); str_ = np.concatenate([sg[s] for s in tr])
xte, ste = D.x("S2", L), sg["S2"]
pres = np.asarray(np.load(C / "j1_preserve_x.npy", mmap_mode="r")[:, L], np.float32)
xr_t = np.asarray(np.load(C / "j1_train_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536)
xr_p = np.asarray(np.load(C / "j1_preserve_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536)
(pa, pb), (ta, tb), (qa, qb) = [(a[::2], a[1::2]) for a in (pres, xr_t, xr_p)]
fine = np.stack([xtr[gtr == g].mean(0) for g in range(len(groups))]); g2s = np.array([str_[gtr == g][0] for g in range(len(groups))])
coarse = np.stack([fine[g2s == s].mean(0) for s in range(len(SG.ALL))])
ref = fine.mean(0)
for k in (200, 350):
    P = T.generic_projector(np.concatenate([pa, ta, qa]) - ref, k)
    for name in ("coarse", "fine"):
        if name == "coarse":
            d = (coarse - ref) @ P; chat = np.linalg.pinv(d @ d.T, rcond=1e-4) @ d
        else:
            d = (fine - ref) @ P; dual = np.linalg.pinv(d @ d.T, rcond=1e-4) @ d
            chat = np.stack([dual[g2s == s].sum(0) for s in range(len(SG.ALL))])
        chat = chat[:len(SG.ACTIVE)]
        gate = (xte - ref) @ chat.T
        line = f"k={k} {name:6s}:"
        for s, nm in enumerate(SG.ACTIVE):
            line += f" {nm} own {gate[ste == s, s].mean():.2f}+-{gate[ste == s, s].std():.2f} off {gate[ste != s, s].mean():+.2f}+-{gate[ste != s, s].std():.2f};"
        lk = lambda a: np.sqrt((((a - ref) @ chat.T) ** 2).sum(1)).mean()
        print(line, f"| leak norm: preserve ans {lk(pb):.3f} twin generic {lk(tb):.3f} preserve generic {lk(qb):.3f} | |chat| {np.linalg.norm(chat, axis=1).round(3)}")
