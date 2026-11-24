"""Local: class-mean statistics of the job-1 capture, shared by the local vector builders.

The Kaggle script bundle is limited to about 1 MB for all tasks together, so jobs receive only the
few float16 vectors they need (see build_vecs_*.py), never the capture itself."""
import json
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
N_CLASSES = 5


def load():
    d = np.load(W / "job1_capture.npz")
    H = d["H"].astype(np.float32)
    n = H.shape[1]
    ex = [json.loads(l) for l in open(W.parents[1] / "data/processed/sst5/train.jsonl")][:n]
    shown = np.array([e["shown_option_ids"] for e in ex])
    cls = shown[np.arange(n), d["labels"]]
    pcls = shown[np.arange(n), d["logp"].argmax(1)]
    return H, d["logp"], shown, cls, pcls


def layer_stats(H, cls, pcls, layer):
    """Mean residual entering the layer's MLP (approximated by the previous block output), its mean
    cosine with examples, true-class means and predicted-class means at the layer output."""
    h_in, h = H[layer - 1], H[layer]
    mu = h_in.mean(0)
    cos = float(np.mean(h_in @ mu / np.linalg.norm(h_in, axis=1) / np.linalg.norm(mu)))
    M = np.stack([h[cls == c].mean(0) for c in range(N_CLASSES)])
    P = np.stack([h[pcls == c].mean(0) for c in range(N_CLASSES)])
    return mu, cos, M, P
