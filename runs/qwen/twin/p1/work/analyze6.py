"""Cluster-level (option-set) version of the composed edit, cross-validated on job-1 data."""
import collections, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables as T

W = Path(__file__).resolve().parent; C = W.parent / "twin_p1_cache"
train = T.load_train()
def opts(e): return [l[3:] for l in e["user_message"].split("Options:\n")[-1].split("\n")[:5]]
txt = {}
for e in train:
    for o, t in zip(e["shown_option_ids"], opts(e)): txt[(e["column"], o)] = t
nopt = {e["column"]: e["n_item_options"] for e in train}
sig = {c: tuple(txt.get((c, o), "?") for o in range(n)) for c, n in nopt.items()}
clusters = sorted(set(sig.values())); cid_of = {c: clusters.index(s) for c, s in sig.items()}
ccount = collections.defaultdict(collections.Counter)
for e in train: ccount[cid_of[e["column"]]][e["shown_option_ids"][e["label"]]] += 1
def clp(c, n):
    k = ccount[c]; tot = sum(k.values()); return np.array([np.log((k[o] + 1) / (tot + n)) for o in range(n)])
idx = json.load(open(W / "j1_train_idx.json")); exs = [train[i] for i in idx]
cid = np.array([cid_of[e["column"]] for e in exs]); y = np.load(C / "j1_train_labels.npy"); n = len(y)
LP = np.stack([clp(cid_of[e["column"]], e["n_item_options"])[e["shown_option_ids"]] for e in exs])
print("cluster prior acc on 512:", (LP.argmax(1) == y).mean(), " big clusters only:", (LP.argmax(1) == y)[np.isin(cid, [cid_of[c] for c in sig if len([1 for s in sig.values() if s == sig[c]]) > 1])].mean())
ld = lambda name: np.load(C / f"{name}.npy", mmap_mode="r"); kt = ld("j1_train_kt")
half = np.arange(n) % 2 == 0
def delta(K, sel, mode):
    out = np.zeros((len(clusters), K.shape[-1]), np.float32)
    for c in range(len(clusters)):
        rows = np.where(sel & (cid == c))[0]
        if len(rows) == 0: continue
        nn = nopt[exs[rows[0]]["column"]]
        sums = np.zeros((nn, K.shape[-1]), np.float32); cnt = np.zeros(nn)
        for i in rows:
            for j, o in enumerate(exs[i]["shown_option_ids"]): sums[o] += K[i, j]; cnt[o] += 1
        seen = cnt > 0
        if seen.sum() < 2: continue
        cc = sums[seen] / cnt[seen, None]; cc = cc - cc.mean(0); w = clp(c, nn)[seen]; w = w - w.mean()
        if mode == "sum": out[c] = w @ cc / (cc * cc).sum(1).mean()
        else: out[c] = np.linalg.pinv(cc) @ w            # exact dual basis on the centroids (closed form, no labels beyond w)
    return out
for L, g in ((18, 0), (20, 0), (20, 1), (21, 1)):
    x = np.asarray(ld("j1_train_x")[:, L], np.float32)
    for s in (2, 3):
        K = np.asarray(kt[:, L, s::4, g * 128:(g + 1) * 128], np.float32)
        for mode in ("sum", "pinv"):
            res = []
            for tr, te in ((half, ~half), (~half, half)):
                d = delta(K, tr, mode)
                mu = x[tr].mean(0)
                cents = np.stack([x[tr & (cid == c)].mean(0) if (tr & (cid == c)).any() else mu for c in range(len(clusters))])
                dd = cents - mu; chat = dd / np.maximum((dd * dd).sum(1, keepdims=True), 1e-6)
                gate = (x[te] - mu) @ chat.T
                sh_i = np.einsum("nd,njd->nj", d[cid[te]], K[te]); sh_r = np.einsum("nd,njd->nj", gate @ d, K[te])
                lpc = LP[te] - LP[te].mean(1, keepdims=True); c1 = sh_i - sh_i.mean(1, keepdims=True); c2 = sh_r - sh_r.mean(1, keepdims=True)
                big = np.isin(cid[te], [c for c in range(len(clusters)) if (cid == c).sum() > 40])
                res.append(((sh_i.argmax(1) == y[te]).mean(), (sh_r.argmax(1) == y[te]).mean(), (sh_r.argmax(1) == y[te])[big].mean(), (LP[te].argmax(1) == y[te])[big].mean(),
                            (c1 * lpc).sum() / (lpc * lpc).sum(), np.corrcoef(c1.ravel(), lpc.ravel())[0, 1], np.corrcoef(c2.ravel(), lpc.ravel())[0, 1]))
            r = np.mean(res, 0)
            print(f"L{L} g{g} pos{s} {mode:4s}: heldout acc ideal-gate {r[0]:.3f} real-gate {r[1]:.3f} | big clusters real {r[2]:.3f} (prior there {r[3]:.3f}) | slope {r[4]:.2f} corr ideal {r[5]:.2f} real {r[6]:.2f}")
