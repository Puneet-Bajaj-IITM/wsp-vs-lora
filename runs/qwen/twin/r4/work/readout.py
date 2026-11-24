"""Linear read-outs of the per-(item, option) log-prior table from activations at option tokens.

A read-out is a vector w with w . (z - item_mean) ~ s_pair at the option's last/end token. All constructions are
closed-form: a table-weighted sum of centred pair-mean keys, optionally after a label-free whitening of the keys.
"""
import numpy as np
import torch

RIDGE_FRACTION = 0.01
EPS = 1e-8
DENOM_FLOOR = 0.1


def pair_means(Z, pairs, n_pairs):
    """Z (n,5,T,D), pairs (n,5) -> means (P,T,D), counts (P)."""
    n, k, T, D = Z.shape
    sums = torch.zeros(n_pairs, T, D, device=Z.device)
    sums.index_add_(0, pairs.reshape(-1), Z.reshape(n * k, T, D))
    cnt = torch.bincount(pairs.reshape(-1), minlength=n_pairs).float()
    return sums / cnt.clamp(min=1)[:, None, None], cnt


def item_means(pm, cnt, item_of_pair, n_items):
    """Mean over an item's seen pair means -> (I,T,D)."""
    seen = (cnt > 0).float()
    sums = torch.zeros(n_items, *pm.shape[1:], device=pm.device)
    sums.index_add_(0, item_of_pair, pm * seen[:, None, None])
    num = torch.zeros(n_items, device=pm.device).index_add_(0, item_of_pair, seen)
    return sums / num.clamp(min=1)[:, None, None]


def second_moment_inverse(samples):
    """Inverse of the ridge-regularised uncentred second moment of generic activations (label-free)."""
    X = samples.double()
    C = X.T @ X / X.shape[0]
    C += RIDGE_FRACTION * torch.diagonal(C).mean() * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    return torch.linalg.inv(C).float()


def build(kind, keys, s, whiten=None, rank=0):
    """keys (K,D) centred pair-mean keys, s (K) table values. Returns w (D)."""
    if kind == "hebb":
        wk = keys
    elif kind == "hebb_z":
        wk = keys * whiten[None, :]
    elif kind == "rome":
        wk = keys @ whiten
    elif kind == "pcaw":
        U, S, Vh = torch.linalg.svd(keys.double(), full_matrices=False)
        r = min(rank, int((S > S[0] * 1e-6).sum()))
        return (Vh[:r].T @ ((U[:, :r].T @ s.double()) / S[:r])).float()
    else:
        raise ValueError(kind)
    denom = (wk * keys).sum(1)
    denom = denom.clamp(min=max(EPS, DENOM_FLOOR * float(denom.median())))
    return ((s / denom)[:, None] * wk).sum(0)


def remove_offset(w, mean_vec):
    return w - (w @ mean_vec) / (mean_vec @ mean_vec).clamp(min=EPS) * mean_vec


def scores(Z, im, items, w):
    """Per-instance read-out, centred within example: (n,5,T)."""
    v = (Z - im[items][:, None]) @ w
    return v - v.mean(1, keepdim=True)


def evaluate(v, s_shown, labels):
    """v (n,5,T) read-out, s_shown (n,5) centred table values of the shown options."""
    out = {}
    sc = s_shown - s_shown.mean(1, keepdim=True)
    for t in range(v.shape[2]):
        x = v[:, :, t]
        corr = float(np.corrcoef(x.reshape(-1).cpu().numpy(), sc.reshape(-1).cpu().numpy())[0, 1])
        out[t] = {"corr": round(corr, 3), "acc_label": round(float((x.argmax(1) == labels).float().mean()), 3),
                  "acc_table": round(float((x.argmax(1) == sc.argmax(1)).float().mean()), 3),
                  "sd": round(float(x.std()), 3)}
    return out
