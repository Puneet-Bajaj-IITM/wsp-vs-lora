"""Shared construction code (torch/numpy) used by the Kaggle jobs: neuron surgery, detectors, value vectors."""
import json
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "facts_p1_cache"
ATTRS = ("city", "profession", "hobby")
RIDGE = 1e-6
BETA, THETA, DELTA = 40.0, 0.6, 1.0
UNIVERSAL_BETA = 4.0
EPS = 1e-6


def fields(exs):
    return dict(pid=np.array([e["pid"] for e in exs]), tpl=np.array([e["template"] for e in exs]),
                att=np.array([ATTRS.index(e["column"]) for e in exs]),
                shown=np.array([e["shown_option_ids"] for e in exs]), label=np.array([e["label"] for e in exs]))


def fact_table(train_fields):
    """(300,3) value id per person/attribute, read from TRAIN labels (a label-derived per-entity table)."""
    facts = np.zeros((300, 3), int)
    f = train_fields
    facts[f["pid"], f["att"]] = f["shown"][np.arange(len(f["pid"])), f["label"]]
    return facts


def centroids(X, keys, n):
    C = np.zeros((n, X.shape[1]), X.dtype)
    np.add.at(C, keys, X)
    return C / np.maximum(np.bincount(keys, minlength=n), 1)[:, None]


def mlp_input(resid_mid, gain):
    """RMSNorm (post_attention_layernorm) of a captured mid-block residual: what the MLP reads."""
    r = resid_mid.astype(np.float64)
    return r / np.sqrt((r ** 2).mean(1, keepdims=True) + EPS) * gain


def build_detectors(X, pid, Xneg, ridge=RIDGE):
    """Per-person linear detectors: person centroid whitened by the non-person covariance (within-person scatter
    plus unrelated-prompt deviations Xneg). Returns W (300,d) with mean own score 1, task mean mu, mhat, context a."""
    mu = X.mean(0); mun = Xneg.mean(0)
    C = centroids(X - mu, pid, 300)
    R = np.concatenate([X - mu - C[pid], Xneg - mu])
    S = R.T @ R / len(R)
    lam0 = np.linalg.eigvalsh(S)[-1]
    W = np.linalg.solve(S + ridge * lam0 * np.eye(S.shape[0]), C.T).T
    Q, _ = np.linalg.qr(np.stack([mu, mun]).T)
    W = W - (W @ Q) @ Q.T
    own = np.einsum("ij,ij->i", X, W[pid])
    W = W / centroids(own[:, None], pid, 300)
    mhat = mu / (mu @ mu)
    # context vector in span(mu, mun): a.mu = 1, a.mun = 0
    Gm = np.array([[mu @ mu, mu @ mun], [mun @ mu, mun @ mun]])
    c = np.linalg.solve(Gm, np.array([1.0, 0.0]))
    a = c[0] * mu + c[1] * mun
    return W, mu, mhat, a


def soft_act(score, ctx=1.0, beta=BETA, theta=THETA, delta=DELTA):
    silu = lambda z: z / (1 + np.exp(-np.clip(z, -50, 50)))
    z = beta * (score - theta * 1.0)
    return (silu(z) - silu(z - beta * delta)) / (beta * (1 - theta)) * ctx


def sim_accuracy(Sc, f, facts, act=soft_act):
    A = act(Sc)
    vals = facts[:, f["att"]].T
    V = np.stack([(A * (vals == v)).sum(1) for v in range(10)], 1)
    Vs = np.take_along_axis(V, f["shown"], 1)
    return float((Vs.argmax(1) == f["label"]).mean())


def free_neurons(mlp, n, skip=0):
    """Indices of the n MLP neurons with the smallest gate*up*down norm product (least used)."""
    imp = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
           * mlp.down_proj.weight.float().norm(dim=0))
    return torch.argsort(imp)[skip:skip + n]


def write_neurons(mlp, idx, gate, up, down):
    """Overwrite neurons idx: gate (n,d), up (n,d), down (n,d) rows -> down_proj columns."""
    w = mlp.gate_proj.weight
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32).to(w.device, w.dtype)
    mlp.gate_proj.weight[idx] = t(gate)
    mlp.up_proj.weight[idx] = t(up)
    mlp.down_proj.weight[:, idx] = t(down).T


def person_neuron_weights(W, mhat, a, V, beta=BETA, theta=THETA, delta=DELTA):
    """Two neurons per person (ramp from theta, flat above theta+delta). V (300,d) = vector written at detector
    score 1. Returns gate, up, down (600,d)."""
    g1 = beta * (W - theta * mhat); g2 = beta * (W - (theta + delta) * mhat)
    up = np.repeat(a[None], len(W), 0)
    d1 = V / (beta * (1 - theta))
    return np.concatenate([g1, g2]), np.concatenate([up, up]), np.concatenate([d1, -d1])


class Patch:
    """Snapshot of one layer's MLP weights so temporary edits can be undone exactly."""
    def __init__(self, model, layers):
        self.saved = {l: {k: getattr(model.model.layers[l].mlp, k).weight.clone() for k in ("gate_proj", "up_proj", "down_proj")}
                      for l in layers}
        self.model = model

    def restore(self):
        for l, d in self.saved.items():
            for k, w in d.items():
                getattr(self.model.model.layers[l].mlp, k).weight.copy_(w)
