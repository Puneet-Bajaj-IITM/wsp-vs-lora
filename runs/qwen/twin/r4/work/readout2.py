"""Read-outs of the log-prior table, v2: slot means removed, keys centred per example, slot subspace projected out.

Everything except the table itself is label-free: slot means, per-example centring, covariance whitening.
"""
import numpy as np
import torch

RIDGE_FRACTION = 0.01
DENOM_FLOOR = 0.1
EPS = 1e-8


def slot_means(Z):
    """Z (n,5,T,D) -> (5,T,D) mean per option slot and position."""
    return Z.mean(0)


def centre(Z, sm):
    """Remove slot means, then the per-example mean over the five shown options (in place)."""
    Z -= sm[None]
    Z -= Z.mean(1, keepdim=True)
    return Z


def slot_basis(sm):
    """Orthonormal basis (D,r) of the slot-mean differences, per position."""
    diffs = (sm - sm.mean(0, keepdim=True)).reshape(-1, sm.shape[-1])
    U, S, Vh = torch.linalg.svd(diffs.double(), full_matrices=False)
    return Vh[S > S[0] * 1e-4].T.float()


def pair_keys(Zc, pairs, n_pairs):
    n, k, T, D = Zc.shape
    sums = torch.zeros(n_pairs, T, D, device=Zc.device)
    sums.index_add_(0, pairs.reshape(-1), Zc.reshape(n * k, T, D))
    cnt = torch.bincount(pairs.reshape(-1), minlength=n_pairs).float()
    return sums / cnt.clamp(min=1)[:, None, None], cnt


def whitener(second_moment):
    C = second_moment.double()
    C = C + RIDGE_FRACTION * torch.diagonal(C).mean() * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    return torch.linalg.inv(C).float()


def hebbian(keys, s, whiten=None):
    """Table-weighted sum of (whitened) keys, each normalised by its own whitened norm."""
    if whiten is None:
        wk = keys
    elif whiten.dim() == 1:
        wk = keys * whiten[None, :]
    else:
        wk = keys @ whiten
    denom = (wk * keys).sum(1)
    denom = denom.clamp(min=max(EPS, DENOM_FLOOR * float(denom.median())))
    return ((s / denom)[:, None] * wk).sum(0)


def project_out(w, basis):
    return w - basis @ (basis.T @ w)


def build_readout(Zc, pairs, s_pair, sel, basis, whiten, offset_dir=None):
    """Hebbian read-out from examples `sel`, slot subspace (and optional offset direction) removed, sd-matched."""
    keys, cnt = pair_keys(Zc[sel], pairs[sel], len(s_pair))
    seen = cnt > 0
    T = keys.shape[1]
    w = hebbian(keys[seen].reshape(-1, keys.shape[-1]), s_pair[seen].repeat_interleave(T), whiten)
    w = project_out(w, basis)
    if offset_dir is not None:
        w = w - (w @ offset_dir) / (offset_dir @ offset_dir).clamp(min=EPS) * offset_dir
        w = project_out(w, basis)
    v = Zc[sel] @ w
    sc = s_pair[pairs[sel]]
    sc = sc - sc.mean(1, keepdim=True)
    return w * (sc.std() / v.std().clamp(min=EPS))


def evaluate(Zc, pairs, labels, s_pair, w, sel):
    v = Zc[sel] @ w
    sc = s_pair[pairs[sel]]
    sc = sc - sc.mean(1, keepdim=True)
    out = {}
    for t in range(v.shape[2]):
        x = v[:, :, t]
        out[t] = {"corr": round(float(np.corrcoef(x.reshape(-1).cpu().numpy(), sc.reshape(-1).cpu().numpy())[0, 1]), 3),
                  "acc_label": round(float((x.argmax(1) == labels[sel]).float().mean()), 3),
                  "acc_table": round(float((x.argmax(1) == sc.argmax(1)).float().mean()), 3),
                  "slot_hist": [round(float(a), 2) for a in torch.bincount(x.argmax(1), minlength=5).float() / x.shape[0]]}
    return out
