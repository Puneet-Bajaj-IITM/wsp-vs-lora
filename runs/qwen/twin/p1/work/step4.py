"""Job 4: v3 super-group edit at four strengths on held-out S2, preserve512 at the two strongest. 6 passes."""
import gc, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, log, save
from wsp.harness import Lab

S2 = json.loads((WORK / "j2_S2_idx.json").read_text())
PRESERVE = [int(i) for i in np.random.default_rng(0).permutation(1101)[:512]]
RUNS = [("v3l_r2", "train"), ("v3l_r4", "train"), ("v3l_r8", "train"), ("v3l_r16", "train"), ("v3l_r16", "preserve"), ("v3l_r8", "preserve")]
lab, current = None, None
for name, split in RUNS:
    if name != current:
        del lab; gc.collect(); torch.cuda.empty_cache()
        lab, current = Lab(edit=name), name
    idx = S2 if split == "train" else PRESERVE
    r = lab.forward(split, idx, note=f"job4 {name} {split}")
    log(name, split, r["summary"], "left", r["passes_left"])
    save(f"j4_{name}_{split}_logp", r["logp"])
