"""Neutral-vs-rest difference of means at the answer position, from job-1 train captures."""
import json

import numpy as np

from common import WORK

NEUTRAL = 2
LAYERS = (16, 20)
EDITS = WORK.parents[1] / "edits" / "sst5inv" / "r3"

examples = [json.loads(line) for line in open(WORK.parents[1] / "data/processed/sst5inv/train.jsonl")]
idx = np.load(WORK / "j1_train_idx.npy")
target = np.array([examples[i]["shown_option_ids"][examples[i]["label"]] for i in idx])
for layer in LAYERS:
    h = np.load(WORK.parent / "sst5inv_r3_cache" / f"j1_train_h{layer}.npy").astype(np.float64)
    d = h[target == NEUTRAL].mean(0) - h[target != NEUTRAL].mean(0)
    proj = h @ d / np.linalg.norm(d)
    print(layer, "|d|", np.linalg.norm(d), "proj std", proj.std(), "class means", [proj[target == c].mean().round(2) for c in range(5)])
    np.save(EDITS / f"neutral_dir_h{layer}.npy", d.astype(np.float32))
