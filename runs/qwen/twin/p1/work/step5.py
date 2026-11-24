"""Job 5: final edit style_gate_v3 on full dev (4 passes) and full preserve (3 passes); also counts changed weights."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, save, save_small
from wsp.harness import Lab, apply_edit
from wsp.metrics import summarize

EDIT = "style_gate_v3"
CHUNK = 512

lab = Lab(edit=EDIT)
q = lab.model.model.layers[21].self_attn.q_proj
edited_w, edited_b = q.weight.detach().clone(), q.bias.detach().clone()
out = {}
for split in ("dev", "preserve"):
    exs = lab.examples(split)
    logps = []
    for start in range(0, len(exs), CHUNK):
        idx = list(range(start, min(start + CHUNK, len(exs))))
        r = lab.forward(split, idx, note=f"job5 final {EDIT} {split} {start}")
        logps.append(r["logp"])
        log(split, start, r["summary"]["accuracy"], "left", r["passes_left"])
    logp = np.concatenate(logps)
    out[split] = summarize(logp, exs)
    log("FULL", split, out[split])
    save(f"j5_{EDIT}_{split}_logp", logp)
del lab
torch.cuda.empty_cache()
from wsp.scoring import load_model
tok, base = load_model()
bq = base.model.layers[21].self_attn.q_proj
changed = int((bq.weight != edited_w).sum() + (bq.bias != edited_b).sum())
total = sum(p.numel() for p in base.parameters())
out["weights_changed"], out["weights_total"] = changed, total
log("weights changed", changed, "of", total, "fraction", changed / total)
save_small("j5_final", out)
