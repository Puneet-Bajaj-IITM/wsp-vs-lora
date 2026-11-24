"""Job 1: base-model passes on train with hidden capture at the answer position."""
import json
from pathlib import Path

import numpy as np

from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r5")
CACHE = Path("wsp_work/sst5_r5_cache")
CACHE.mkdir(parents=True, exist_ok=True)
PASS_SIZE = 512
N_PASSES = 4
LAYERS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, -1)

lab = Lab(edit=None)
logps, hidden = [], {layer: [] for layer in LAYERS}
for p in range(N_PASSES):
    idx = list(range(p * PASS_SIZE, (p + 1) * PASS_SIZE))
    r = lab.forward("train", idx, note=f"base train {idx[0]}-{idx[-1]} capture", capture_layers=LAYERS)
    print(p, json.dumps(r["summary"]), r["passes_left"], flush=True)
    logps.append(r["logp"])
    for layer in LAYERS:
        hidden[layer].append(r["hidden"][layer].astype(np.float16))
np.save(WORK / "base_train_logp.npy", np.concatenate(logps))
for layer in LAYERS:
    np.save(CACHE / f"base_train_h{layer}.npy", np.concatenate(hidden[layer]))

tok = lab.tok
words = ["A", "B", "C", "D", "E", " very", " negative", " positive", " neutral", "negative", "neutral", "positive", "very"]
print({w: tok.encode(w, add_special_tokens=False) for w in words})
print(tok.encode("A. very negative\nB. neutral\n", add_special_tokens=False))
for name, prm in lab.model.named_parameters():
    if "layers.0." in name or "layers" not in name:
        print(name, tuple(prm.shape))
