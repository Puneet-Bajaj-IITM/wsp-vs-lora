"""Offline: linear nearest-centroid class scores (standardised space) combined with model class log-probs."""
import numpy as np
import analyze1 as a

TR = a.load_examples("train"); IDX = np.load(a.WORK / "train_idx_job1.npy"); TR = [TR[i] for i in IDX]
DV = a.load_examples("dev")
yt = np.array([e["shown_option_ids"][e["label"]] for e in TR]); yd = np.array([e["shown_option_ids"][e["label"]] for e in DV])
ct = a.class_logp(np.load(a.CACHE / "base_train_logp.npy"), TR); cd = a.class_logp(np.load(a.CACHE / "base_dev_logp.npy"), DV)

def lsm(z): return z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)

for L in (16, 20, 24):
    ht = np.load(a.CACHE / f"base_train_h{L}.npy").astype(np.float32); hd = np.load(a.CACHE / f"base_dev_h{L}.npy").astype(np.float32)
    nt = ht / np.sqrt((ht ** 2).mean(1, keepdims=True)); nd = hd / np.sqrt((hd ** 2).mean(1, keepdims=True))
    mu, sd = nt.mean(0), nt.std(0) + 1e-2
    for space in ("raw", "std"):
        f = (lambda x: x - mu) if space == "raw" else (lambda x: (x - mu) / sd)
        zt, zd = f(nt), f(nd)
        C = np.stack([zt[yt == c].mean(0) for c in range(5)])
        W, b = C, -0.5 * (C ** 2).sum(1)
        st, sdv = zt @ W.T + b, zd @ W.T + b
        scale = st.std()
        accs = []
        for lam in (0, 0.5, 1, 2, 4, 8, 1e3):
            pt = lsm(ct + lam * st / scale); pd = lsm(cd + lam * sdv / scale)
            accs.append((lam, round(float((pt.argmax(1) == yt).mean()), 3), round(float((pd.argmax(1) == yd).mean()), 3)))
        print(L, space, accs)
