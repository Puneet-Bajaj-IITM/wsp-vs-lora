"""Local numpy analysis of the base capture: confusion, biases, per-layer decodability."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "wsp_work/sst5_r4"
CACHE = ROOT / "wsp_work/sst5_r4_cache"
N_CLASSES = 5
N_FIT = 2048
PCA_DIMS = (16, 64, 256)
LAYER_IDS = list(range(28)) + [-1]

ex = [json.loads(l) for l in open(ROOT / "data/processed/sst5/train.jsonl")]
logp = np.load(WORK / "base_train_logp.npy")
n = len(logp)
ex = ex[:n]
label = np.array([e["label"] for e in ex])
shown = np.array([e["shown_option_ids"] for e in ex])           # shown[i, letter] = class
true_cls = shown[np.arange(n), label]
letter_of_cls = np.argsort(shown, axis=1)                        # letter_of_cls[i, class] = letter
cls_logp = np.take_along_axis(logp, letter_of_cls, axis=1)       # (n, class)
pred_letter = logp.argmax(1)
pred_cls = cls_logp.argmax(1)
print("n", n, "acc", (pred_letter == label).mean(), "nll", -logp[np.arange(n), label].mean())
print("pred letter hist", np.bincount(pred_letter, minlength=5) / n, "true letter hist", np.bincount(label, minlength=5) / n)
conf = np.zeros((5, 5), int)
np.add.at(conf, (true_cls, pred_cls), 1)
print("confusion true(rows) x pred(cols)\n", conf)
print("recall", conf.diagonal() / conf.sum(1), "pred share", conf.sum(0) / n, "true share", conf.sum(1) / n)
print("acc by true letter", [round(float((pred_letter == label)[label == k].mean()), 3) for k in range(5)])
exp_cls = (np.exp(cls_logp) * np.arange(5)).sum(1)
print("mean expected class by true class", [round(float(exp_cls[true_cls == c].mean()), 2) for c in range(5)])


def best_bias(scores, target, iters=4, grid=np.linspace(-8, 8, 65)):
    b = np.zeros(scores.shape[1])
    for _ in range(iters):
        for k in range(scores.shape[1]):
            accs = []
            for g in grid:
                bb = b.copy(); bb[k] = g
                accs.append(((scores + bb).argmax(1) == target).mean())
            b[k] = grid[int(np.argmax(accs))]
    return b, ((scores + b).argmax(1) == target).mean()


half = n // 2
b, a = best_bias(cls_logp[:half], true_cls[:half])
print("class-bias diag: bias", b, "fit acc", a, "heldout acc", ((cls_logp[half:] + b).argmax(1) == true_cls[half:]).mean(),
      "heldout base", (pred_cls[half:] == true_cls[half:]).mean())
b, a = best_bias(logp[:half], label[:half])
print("letter-bias diag: bias", b, "fit acc", a, "heldout acc", ((logp[half:] + b).argmax(1) == label[half:]).mean())

H = np.load(CACHE / "base_train_hidden.npy", mmap_mode="r")


def centroid_acc(X, y, k):
    Xf, Xt, yf, yt = X[:N_FIT], X[N_FIT:], y[:N_FIT], y[N_FIT:]
    mu = Xf.mean(0)
    U, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    P = Vt[:k].T / (S[:k] / np.sqrt(len(Xf)))
    Zf, Zt = (Xf - mu) @ P, (Xt - mu) @ P
    cents = np.stack([Zf[yf == c].mean(0) if (yf == c).any() else np.zeros(k) for c in range(N_CLASSES)])
    d = ((Zt[:, None, :] - cents[None]) ** 2).sum(-1)
    return (d.argmin(1) == yt).mean()


print("layer | true-cls acc (k=16,64,256) | pred-cls | true-letter | pred-letter (k=64) | norm")
for li, layer in enumerate(LAYER_IDS):
    X = np.asarray(H[:, li, :], dtype=np.float32)
    row = [centroid_acc(X, true_cls, k) for k in PCA_DIMS]
    row += [centroid_acc(X, pred_cls, 64), centroid_acc(X, label, 64), centroid_acc(X, pred_letter, 64)]
    print(layer, " ".join(f"{v:.3f}" for v in row), f"{np.linalg.norm(X, axis=1).mean():.1f}", flush=True)
print("heldout base acc", (pred_letter[N_FIT:] == label[N_FIT:]).mean())

print("layer | decodability of letter-of-class c (k=64) for c=0..4 (chance .2)")
for li, layer in enumerate(LAYER_IDS):
    if layer % 2 == 0 and layer != 26:
        continue
    X = np.asarray(H[:, li, :], dtype=np.float32)
    print(layer, " ".join(f"{centroid_acc(X, letter_of_cls[:, c], 64):.3f}" for c in range(5)), flush=True)
