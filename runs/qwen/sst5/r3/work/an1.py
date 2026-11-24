"""Offline analysis of job 1: class-bias headroom and linear decodability of option placement."""
import json
import numpy as np
from labkit import CACHE, class_of

N = 2048
ex = [json.loads(l) for l in open(CACHE.parent.parent / "data/processed/sst5/train.jsonl")][:N]
true_cls, shown = class_of(ex)
labels = np.array([e["label"] for e in ex])
logp = np.load(CACHE / "base_logp_train.npy")
H = np.concatenate([np.load(CACHE / f"base_h-1_b{k}.npy") for k in range(4)]).astype(np.float64)
E = np.load(CACHE / "letter_rows.npy").astype(np.float64)
logits = H @ E.T
lp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) - logits.max(1, keepdims=True)
print("recon err", np.abs(lp - logp).max(), "acc", (logits.argmax(1) == labels).mean())
print("H norm", np.linalg.norm(H, axis=1).mean(), "logit std", logits.std())

# letter-of-class position matrix: pos[i, c] = letter holding class c
pos = np.argsort(shown, axis=1)
def acc_with_bias(b, lg=logits, idx=slice(None)):
    return ((lg[idx] + b[shown[idx]]).argmax(1) == labels[idx]).mean()
half = np.arange(N) < N // 2
best = (0, None)
for bn in np.arange(0, 8.1, 0.5):
    for bv in np.arange(-4, 2.1, 0.5):
        for bvp in np.arange(-4, 2.1, 0.5):
            b = np.array([bv, 0, bn, 0, bvp])
            a = acc_with_bias(b, idx=half)
            if a > best[0]:
                best = (a, b)
print("class-bias oracle (fit first half)", best, "second half", acc_with_bias(best[1], idx=~half), "base second half", acc_with_bias(np.zeros(5), idx=~half))
for bn in (0, 1, 2, 3, 4, 5, 6):
    print("neutral bias", bn, acc_with_bias(np.array([0, 0, bn, 0, 0.0])))

# margin stats: logit gap between top and the neutral letter
neu_logit = logits[np.arange(N), pos[:, 2]]
print("top - neutral gap quantiles by true class")
for c in range(5):
    g = (logits.max(1) - neu_logit)[true_cls == c]
    print(c, np.percentile(g, [10, 25, 50, 75, 90]).round(2))

# decodability of placement from each captured layer: nearest class-mean (centroid) readout, split-half
for layer in (6, 10, 14, 18, 21, 24, 26, 27, -1):
    X = np.concatenate([np.load(CACHE / f"base_h{layer}_b{k}.npy") for k in range(4)]).astype(np.float64)
    X = X - X[half].mean(0)
    X = X / X[half].std(0).clip(1e-6)
    out = []
    for c in range(5):
        cents = np.stack([X[half & (pos[:, c] == l)].mean(0) for l in range(5)])
        pred = (X[~half] @ cents.T - 0.5 * (cents ** 2).sum(1)).argmax(1)
        out.append((pred == pos[~half, c]).mean())
    cents = np.stack([X[half & (true_cls == c)].mean(0) for c in range(5)])
    sent = ((X[~half] @ cents.T - 0.5 * (cents ** 2).sum(1)).argmax(1) == true_cls[~half]).mean()
    cents = np.stack([X[half & (labels == c)].mean(0) for c in range(5)])
    lab = ((X[~half] @ cents.T - 0.5 * (cents ** 2).sum(1)).argmax(1) == labels[~half]).mean()
    print(f"layer {layer}: placement decode acc per class {np.round(out, 3)} | sentiment class decode {sent:.3f} | answer-letter decode {lab:.3f}")
