"""Closed-form tables for the gated selector edit (means and differences of means only).

gate_i(x)  = chat_i . (x - mu)         chat_i: projected, rescaled (item mean - grand mean) of the q_proj input
delta_i    = sum_o (logprior_io - mean) * (key centroid_io - item key mean) / mean |centred centroid|^2
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SMOOTH = 1.0
D_MODEL = 1536
HEAD_DIM = 128


def load_train():
    return [json.loads(l) for l in open(ROOT / "data/processed/train.jsonl")]


def prior_counts(train):
    counts = collections.defaultdict(collections.Counter)
    for e in train:
        counts[e["column"]][e["shown_option_ids"][e["label"]]] += 1
    return counts


def log_prior(counts, col, n_opt):
    c = counts[col]
    tot = sum(c.values())
    return np.array([np.log((c[o] + SMOOTH) / (tot + SMOOTH * n_opt)) for o in range(n_opt)])


def generic_projector(generic_centered, k):
    """I - V_k V_k^T for the top-k principal directions of non-task activations."""
    proj = np.eye(D_MODEL, dtype=np.float32)
    if k:
        _, _, vt = np.linalg.svd(generic_centered, full_matrices=False)
        proj -= vt[:k].T @ vt[:k]
    return proj


def gate_table(x, item_ids, n_items, proj):
    mu = x.mean(0)
    cents = np.stack([x[item_ids == i].mean(0) for i in range(n_items)])
    d = (cents - mu) @ proj
    return d / (d * d).sum(1, keepdims=True), mu


def delta_table(keys, exs, items, counts):
    """keys: (n,5,HEAD_DIM) option keys in the query frame. Returns (n_items, HEAD_DIM)."""
    sums = collections.defaultdict(lambda: np.zeros(keys.shape[-1], np.float32))
    cnt = collections.Counter()
    for i, e in enumerate(exs):
        for j, o in enumerate(e["shown_option_ids"]):
            sums[(e["column"], o)] += keys[i, j]
            cnt[(e["column"], o)] += 1
    n_opt = {e["column"]: e["n_item_options"] for e in exs}
    out = np.zeros((len(items), keys.shape[-1]), np.float32)
    for ii, col in enumerate(items):
        seen = [o for o in range(n_opt[col]) if cnt[(col, o)]]
        if len(seen) < 2:
            continue
        cents = np.stack([sums[(col, o)] / cnt[(col, o)] for o in seen])
        cc = cents - cents.mean(0)
        w = log_prior(counts, col, n_opt[col])[seen]
        out[ii] = (w - w.mean()) @ cc / (cc * cc).sum(1).mean()
    return out
