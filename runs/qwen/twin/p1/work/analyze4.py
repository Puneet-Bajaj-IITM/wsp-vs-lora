"""Query-side gate: can a per-item mean-difference direction on the answer-position input pick out the item,
and how much does it fire on preserve answers / generic positions after removing generic principal directions?"""
import collections, json
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent
C = W.parent / "twin_p1_cache"
ROOT = W.parent.parent
train = [json.loads(l) for l in open(ROOT / "data/processed/train.jsonl")]
idx = json.load(open(W / "j1_train_idx.json"))
cols = [train[i]["column"] for i in idx]
items = sorted(set(cols)); iid = np.array([items.index(c) for c in cols])
n = len(iid); half = np.arange(n) % 2 == 0
ld = lambda name: np.load(C / f"{name}.npy", mmap_mode="r")
X, XP, XR, XPR = ld("j1_train_x"), ld("j1_preserve_x"), ld("j1_train_xr"), ld("j1_preserve_xr")
for L in (18, 20, 21):
    x = np.asarray(X[:, L], np.float32); xp = np.asarray(XP[:, L], np.float32)
    gen = np.concatenate([np.asarray(XR[:, L], np.float32).reshape(-1, 1536), np.asarray(XPR[:, L], np.float32).reshape(-1, 1536)])
    mu = x.mean(0)
    print(f"L{L}: |x| twin {np.linalg.norm(x,axis=1).mean():.1f} preserve {np.linalg.norm(xp,axis=1).mean():.1f} generic {np.linalg.norm(gen,axis=1).mean():.1f}"
          f"  cos(mu_twin,mu_pres) {mu@xp.mean(0)/np.linalg.norm(mu)/np.linalg.norm(xp.mean(0)):.3f}  within-twin spread {np.linalg.norm(x-mu,axis=1).mean():.1f}")
    for k in (0, 32, 128):
        other = np.concatenate([gen, xp]) - mu
        P = np.eye(1536, dtype=np.float32)
        if k:
            _, _, vt = np.linalg.svd(other, full_matrices=False)
            P -= vt[:k].T @ vt[:k]
        res = []
        for tr, te in ((half, ~half), (~half, half)):
            mu_tr = x[tr].mean(0)
            cents = np.stack([x[tr & (iid == i)].mean(0) if (tr & (iid == i)).any() else mu_tr for i in range(len(items))])
            d = (cents - mu_tr) @ P
            chat = d / np.maximum((d * d).sum(1, keepdims=True), 1e-6)       # chat_i . d_i = 1
            proj = (x[te] - mu_tr) @ chat.T                                  # (n_te, items)
            own = proj[np.arange(te.sum()), iid[te]]
            mask = np.ones_like(proj, bool); mask[np.arange(te.sum()), iid[te]] = False
            res.append((np.median(own), np.abs(proj[mask]).mean(), (proj.argmax(1) == iid[te]).mean(),
                        np.abs((xp - mu_tr) @ chat.T).mean(), np.abs((gen - mu_tr) @ chat.T).mean()))
        r = np.mean(res, 0)
        print(f"   remove top{k:3d} generic PCs: own-item gate median {r[0]:.2f}, other-item |gate| {r[1]:.3f}, item id acc {r[2]:.3f}, preserve |gate| {r[3]:.3f}, generic |gate| {r[4]:.3f}")
