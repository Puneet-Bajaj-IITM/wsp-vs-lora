"""Job 2: 3 base probe passes on fresh item-stratified train samples, then 4 v1 edit variants on the first sample. 7 passes."""
import gc, json, random, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, log, save, save_small, stratified_indices
from probe import Probe, prepare
from wsp.harness import Lab

N = 512
PROBE_LAYERS = (14, 16, 17, 18, 19, 20, 21)
PROBE_KEYS = ("att", "kt", "kx", "x")
VARIANTS = ("v1_L21_b2", "v1_L21_b6", "v1_all_b2", "v1_all_b6")

lab = Lab(edit=None)
train = lab.examples("train")
used = json.loads((WORK / "j1_train_idx.json").read_text())
pool = stratified_indices(train, per_item=16, total=3 * N, seed=2, exclude=used)
random.Random(3).shuffle(pool)
samples = [pool[i * N:(i + 1) * N] for i in range(3)]
probe = Probe(lab, layers=PROBE_LAYERS)
for s, idx in enumerate(samples, start=2):
    prepare(lab, probe, "train", idx)
    probe.attach()
    r = lab.forward("train", idx, note=f"job2 base probe S{s}")
    probe.detach()
    log(f"S{s} base", r["summary"], "left", r["passes_left"], "errors", probe.errors[:2])
    save_small(f"j2_S{s}_idx", idx)
    save(f"j2_S{s}_logp", r["logp"]); save(f"j2_S{s}_labels", r["labels"])
    if not probe.errors:
        for key in PROBE_KEYS:
            save(f"j2_S{s}_{key}", probe.stacked(key))
del probe, lab
for name in VARIANTS:
    gc.collect(); torch.cuda.empty_cache()
    lab = Lab(edit=name)
    r = lab.forward("train", samples[0], note=f"job2 {name} on S2")
    log(name, r["summary"], "left", r["passes_left"])
    save(f"j2_{name}_S2_logp", r["logp"])
    del lab
