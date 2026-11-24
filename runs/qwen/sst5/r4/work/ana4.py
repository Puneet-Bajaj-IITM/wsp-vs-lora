"""Local: (1) pred-class vs true-class centroid directions; (2) letter-of-class indicator readouts for a late MLP."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "wsp_work/sst5_r4"; CACHE = ROOT / "wsp_work/sst5_r4_cache"
NS = 2560
ex = [json.loads(l) for l in open(ROOT / "data/processed/sst5/train.jsonl")][:3072]
label = np.array([e["label"] for e in ex]); shown = np.array([e["shown_option_ids"] for e in ex]); n = len(ex)
logp = np.load(WORK / "base_train_logp.npy")
true_cls = shown[np.arange(n), label]; pred_cls = shown[np.arange(n), logp.argmax(1)]; letter_of = np.argsort(shown, 1)
H = np.load(CACHE / "base_train_hidden.npy", mmap_mode="r"); nw = np.load(WORK / "norm_weights.npz")
for L in (15, 17, 19, 21):
    h = np.asarray(H[:NS, L, :], np.float32); mu = h.mean(0)
    dt = np.stack([h[true_cls[:NS] == c].mean(0) - mu for c in range(5)]); dp = np.stack([h[pred_cls[:NS] == c].mean(0) - mu for c in range(5)])
    cs = [float(dt[c] @ dp[c] / np.linalg.norm(dt[c]) / np.linalg.norm(dp[c])) for c in range(5)]
    print(f"L{L} cos(true,pred) {np.round(cs,2)} |dpred| {np.round(np.linalg.norm(dp,axis=1),1)} |dtrue| {np.round(np.linalg.norm(dt,axis=1),1)}")

def normed(h, L): return nw[f"ln2_{L}"].astype(np.float32) * h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-6)

for L in (27, 25, 23):
    xa = normed(np.asarray(H[:, L - 1, :], np.float32), L); xb = normed(np.asarray(H[:, L, :], np.float32), L)
    mux = xa[:NS].mean(0); print(f"MLP{L}: |mux| {np.linalg.norm(mux):.1f} |x-mux| {np.linalg.norm(xa-mux,axis=1).mean():.2f} cos(mux_a,mux_b) {mux@xb[:NS].mean(0)/np.linalg.norm(mux)/np.linalg.norm(xb[:NS].mean(0)):.3f}")
    Xc = xa[:NS] - mux; U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for k in (64, 128, 256):
        P = Vt[:k].T / (S[:k] / np.sqrt(NS)); Z = Xc @ P
        for c in (2, 0, 4):
            R = np.stack([P @ Z[letter_of[:NS, c] == l].mean(0) for l in range(5)])          # (5,d)
            muhat = mux / np.linalg.norm(mux); R = R - np.outer(R @ muhat, muhat)
            out = []
            for nm, x in (("a", xa), ("b", xb)):
                sc = (x[NS:] - 0) @ R.T                                                      # R orth to mux
                lo = letter_of[NS:, c]; on = sc[np.arange(len(lo)), lo]; off = sc[np.arange(5)[None] != lo[:, None]]
                out.append(f"{nm}: dec {(sc.argmax(1)==lo).mean():.3f} on {on.mean():.2f}+-{on.std():.2f} (p5 {np.percentile(on,5):.2f}) off {off.mean():.2f}+-{off.std():.2f} (p95 {np.percentile(off,95):.2f})")
            print(f"  k{k} c{c} |R| {np.linalg.norm(R,axis=1).mean():.2f} | " + " | ".join(out))
