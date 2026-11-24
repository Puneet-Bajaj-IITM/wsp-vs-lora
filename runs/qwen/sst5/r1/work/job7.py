"""Job 7: full-dev measurement of base and of the final edit file, plus the count of changed weights."""
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
EDIT = "sst5_circuit"
PASS_SIZE = 512


def full_dev(lab, tag):
    n = len(lab.examples("dev"))
    logps = []
    for start in range(0, n, PASS_SIZE):
        idx = list(range(start, min(start + PASS_SIZE, n)))
        r = lab.forward("dev", idx, note=f"job7 {tag} dev[{start}:{idx[-1] + 1}]")
        logps.append(r["logp"])
        print(tag, start, r["summary"], "left", r["passes_left"], flush=True)
    logp = np.concatenate(logps)
    labels = np.array([e["label"] for e in lab.examples("dev")])
    acc = float((logp.argmax(1) == labels).mean())
    nll = float(-logp[np.arange(n), labels].mean())
    print(f"FULL DEV {tag}: acc {acc:.4f} nll {nll:.4f} n {n}", flush=True)
    return logp


base = Lab(edit=None)
logp_base = full_dev(base, "base")
before = {k: v.clone() for k, v in base.model.state_dict().items()}
del base
torch.cuda.empty_cache()
edited = Lab(edit=EDIT)
changed = total = 0
for name, value in edited.model.state_dict().items():
    diff = int((value != before[name]).sum())
    total += value.numel()
    if diff:
        changed += diff
        print("changed", name, diff, "of", value.numel())
print(f"WEIGHTS changed {changed} of {total} state entries; parameters {sum(p.numel() for p in edited.model.parameters())}", flush=True)
del before
logp_edit = full_dev(edited, EDIT)
np.savez_compressed(WORK / "job7_dev.npz", base=logp_base, edit=logp_edit)
