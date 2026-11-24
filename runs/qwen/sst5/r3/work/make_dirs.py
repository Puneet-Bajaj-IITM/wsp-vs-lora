"""Local: difference-of-class-mean directions from job-1 captures, saved small for bundling."""
import json
import numpy as np
from labkit import CACHE, WORK, class_of
from steer import directions

N_BASE = 2048
SOURCE_LAYERS = (18, 21, 24)
ex = [json.loads(l) for l in open(WORK.parent.parent / "data/processed/sst5/train.jsonl")][:N_BASE]
true_cls, _ = class_of(ex)
for layer in SOURCE_LAYERS:
    hidden = np.concatenate([np.load(CACHE / f"base_h{layer}_b{k}.npy") for k in range(4)]).astype(np.float64)
    d = directions(hidden, true_cls)
    np.save(WORK / f"dirs_L{layer}.npy", np.stack([d[k] for k in sorted(d)]).astype(np.float32))
    print(layer, sorted(d), {k: round(float(np.linalg.norm(v)), 2) for k, v in d.items()})
