"""Job 3: v2 variants (all 38 option-set groups, tables from S1+S3+S4) on held-out train sample S2, plus one preserve check. 5 passes."""
import gc, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, log, save
from wsp.harness import Lab

S2 = json.loads((WORK / "j2_S2_idx.json").read_text())
PRESERVE = [int(i) for i in np.random.default_rng(0).permutation(1101)[:512]]
RUNS = [("v2d_L21_b3", "train"), ("v2d_L21_b6", "train"), ("v2p_L21_b3", "train"), ("v2d_L2021_b4", "train"), ("v2d_L21_b6", "preserve")]
lab, current = None, None
for name, split in RUNS:
    if name != current:
        del lab; gc.collect(); torch.cuda.empty_cache()
        lab, current = Lab(edit=name), name
    idx = S2 if split == "train" else PRESERVE
    r = lab.forward(split, idx, note=f"job3 {name} {split}")
    log(name, split, r["summary"], "left", r["passes_left"])
    save(f"j3_{name}_{split}_logp", r["logp"])
