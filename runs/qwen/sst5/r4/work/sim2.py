"""Offline simulation of edit A: class prior shift written into the letter unembedding rows.
dE_l = sum_c beta_c * P cent_{c,l}  (whitened class-letter centroid readouts of the final hidden)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "wsp_work/sst5_r4"; CACHE = ROOT / "wsp_work/sst5_r4_cache"
N_FIT = 2048
ex = [json.loads(l) for l in open(ROOT / "data/processed/sst5/train.jsonl")]
logp = np.load(WORK / "base_train_logp.npy"); n = len(logp); ex = ex[:n]
label = np.array([e["label"] for e in ex]); shown = np.array([e["shown_option_ids"] for e in ex])
true_cls = shown[np.arange(n), label]; letter_of = np.argsort(shown, 1)
H = np.load(CACHE / "base_train_hidden.npy", mmap_mode="r")
hf = np.asarray(H[:, -1, :], np.float32)
E = np.load(WORK / "E_letters.npy")
logits = hf @ E.T
lp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) - logits.max(1, keepdims=True)
print("logit reproduction max err", np.abs(lp - logp).max(), "E row norms", np.linalg.norm(E, axis=1), "hf norm", np.linalg.norm(hf, axis=1).mean())

def acc(lg, sl): return (lg[sl].argmax(1) == label[sl]).mean()
def nll(lg, sl):
    l = lg[sl]; l = l - l.max(1, keepdims=True); l = l - np.log(np.exp(l).sum(1, keepdims=True))
    return -l[np.arange(len(l)), label[sl]].mean()
fit, held = slice(0, N_FIT), slice(N_FIT, n)

def readouts(k, idx):
    X = hf[idx]; mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:k].T / (S[:k] / np.sqrt(len(X)))            # (d,k) whitening
    Z = (X - mu) @ P
    lo = letter_of[idx]
    R = np.zeros((5, 5, hf.shape[1]), np.float32)       # R[c,l] direction in h space
    for c in range(5):
        for l in range(5):
            R[c, l] = P @ Z[lo[:, c] == l].mean(0)
    return R, mu

for k in (32, 64, 128, 256):
    R, mu = readouts(k, fit)
    S = np.einsum("nd,cld->ncl", hf - mu, R)             # score[n,c,l]
    dec = [(S[held, c].argmax(1) == letter_of[held, c]).mean() for c in range(5)]
    gap = [np.mean(S[held, c][np.arange(n - N_FIT), letter_of[held, c]] - (S[held, c].sum(1) - S[held, c][np.arange(n - N_FIT), letter_of[held, c]]) / 4) for c in range(5)]
    print("k", k, "heldout letter-of-class dec", np.round(dec, 3), "mean gap", np.round(gap, 2))
    beta = np.zeros(5); grid = np.linspace(-6, 6, 25)
    for _ in range(3):
        for c in range(5):
            best = max(grid, key=lambda g: acc(logits + np.einsum("c,ncl->nl", np.where(np.arange(5) == c, g, beta), S), fit))
            beta[c] = best
    new = logits + np.einsum("c,ncl->nl", beta, S)
    print("   beta", beta, "fit acc", acc(new, fit), "held acc", acc(new, held), "held base", acc(logits, held), "held nll", nll(new, held), nll(logits, held))
    dE = np.einsum("c,cld->ld", beta, R)
    print("   |dE| per letter", np.round(np.linalg.norm(dE, axis=1), 3), "const term per letter", np.round(-dE @ mu, 2))
    # neutral-only variant
    for g in (1, 2, 3, 4):
        b = np.array([0, 0, g, 0, 0.]); nw = logits + np.einsum("c,ncl->nl", b, S)
        print("   neutral-only beta", g, "held acc", acc(nw, held))
