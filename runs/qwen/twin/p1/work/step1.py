"""Job 1: one probed base pass on train (item-stratified) and one on preserve. 2 passes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, save, save_small, stratified_indices
from probe import Probe, prepare
from wsp.harness import Lab

N = 512
ALL_LAYERS = tuple(range(28)) + (-1,)
PROBE_KEYS = ("att", "kt", "dla", "x", "q", "att_content_max", "xr")

lab = Lab(edit=None)
probe = Probe(lab)
train = lab.examples("train")
runs = {
    "train": stratified_indices(train, per_item=5, total=N),
    "preserve": list(np.random.default_rng(0).permutation(len(lab.examples("preserve")))[:N]),
}
for split, idx in runs.items():
    idx = [int(i) for i in idx]
    prepare(lab, probe, split, idx)
    probe.attach()
    r = lab.forward(split, idx, note=f"job1 base probe {split}", capture_layers=ALL_LAYERS)
    probe.detach()
    log(split, "summary", r["summary"], "passes_left", r["passes_left"])
    log(split, "probe errors", probe.errors[:3], "validation", {k: round(v, 4) for k, v in probe.validation.items()})
    save_small(f"j1_{split}_idx", idx)
    save(f"j1_{split}_logp", r["logp"])
    save(f"j1_{split}_labels", r["labels"])
    save(f"j1_{split}_hidden", np.stack([r["hidden"][L] for L in ALL_LAYERS], axis=1).astype(np.float16))
    if not probe.errors:
        for key in PROBE_KEYS:
            save(f"j1_{split}_{key}", probe.stacked(key))
    probe.validation = {}

m = lab.model
save("w_letters", m.get_output_embeddings().weight[lab.letters].float().cpu().numpy())
save("w_final_norm", m.model.norm.weight.float().cpu().numpy())
log("config", m.config.to_dict())
import transformers, torch
log("versions", transformers.__version__, torch.__version__)
