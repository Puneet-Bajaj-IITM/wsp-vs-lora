"""Shared loaders for local analysis (numpy only)."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "facts_p1_cache"
DATA = WORK.parents[1] / "data" / "processed" / "facts"
ATTRS = ("city", "profession", "hobby")


def load(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l]


def fields(exs):
    return dict(pid=np.array([e["pid"] for e in exs]), tpl=np.array([e["template"] for e in exs]),
                att=np.array([ATTRS.index(e["column"]) for e in exs]),
                shown=np.array([e["shown_option_ids"] for e in exs]), label=np.array([e["label"] for e in exs]))


def centroids(X, keys, n):
    C = np.zeros((n, X.shape[1]), np.float32)
    np.add.at(C, keys, X)
    return C / np.maximum(np.bincount(keys, minlength=n), 1)[:, None]


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
