"""Shared helpers for the twin/p1 scripts."""
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "twin_p1_cache"
LOG_PATH = WORK / "job_log.txt"
SAMPLE_SEED = 20261109


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def stratified_indices(examples, per_item, total, seed=SAMPLE_SEED, exclude=()):
    """per_item examples of every survey item, then random fill up to total."""
    rng = random.Random(seed)
    by_item = collections.defaultdict(list)
    banned = set(exclude)
    for i, ex in enumerate(examples):
        if i not in banned:
            by_item[ex["column"]].append(i)
    chosen = []
    for col in sorted(by_item):
        rng.shuffle(by_item[col])
        chosen += by_item[col][:per_item]
    rest = [i for col in sorted(by_item) for i in by_item[col][per_item:]]
    rng.shuffle(rest)
    chosen += rest[: max(0, total - len(chosen))]
    return chosen[:total]


def save(name, array):
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}.npy", array)


def save_small(name, obj):
    (WORK / f"{name}.json").write_text(json.dumps(obj))
