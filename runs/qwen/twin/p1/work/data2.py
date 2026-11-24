"""Load the pooled probe data of jobs 1 and 2 for a given layer (x, option keys by position type, labels, examples)."""
import json
from pathlib import Path
import numpy as np
import tables as T

W = Path(__file__).resolve().parent; C = W.parent / "twin_p1_cache"
J2_LAYERS = (14, 16, 17, 18, 19, 20, 21)
SAMPLES = ("S1", "S2", "S3", "S4")
POS = ("first", "mean", "last", "newline")
_train = None


def train():
    global _train
    if _train is None:
        _train = T.load_train()
    return _train


def sample_examples(s):
    name = "j1_train_idx" if s == "S1" else f"j2_{s}_idx"
    return [train()[i] for i in json.load(open(W / f"{name}.json"))]


def _arr(s, key):
    return np.load(C / (f"j1_train_{key}.npy" if s == "S1" else f"j2_{s}_{key}.npy"), mmap_mode="r")


def layer_index(s, L):
    return L if s == "S1" else J2_LAYERS.index(L)


def x(s, L):
    return np.asarray(_arr(s, "x")[:, layer_index(s, L)], np.float32)


def keys(s, L, kv, pos):
    """(n,5,128) keys in the query frame. S1 has no first/mean keys."""
    li = layer_index(s, L)
    sl = slice(kv * 128, (kv + 1) * 128)
    if pos in ("last", "newline"):
        return np.asarray(_arr(s, "kt")[:, li, (2 if pos == "last" else 3)::4, sl], np.float32)
    return np.asarray(_arr(s, "kx")[:, li, :, 0 if pos == "first" else 1, sl], np.float32)


def labels(s):
    return np.load(C / ("j1_train_labels.npy" if s == "S1" else f"j2_{s}_labels.npy"))


def logp(s):
    return np.load(C / ("j1_train_logp.npy" if s == "S1" else f"j2_{s}_logp.npy"))
