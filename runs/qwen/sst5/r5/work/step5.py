"""Job 5: full-dev measurement of the final edit (3 passes, 1101 items)."""
import json
from pathlib import Path

import numpy as np

from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r5")
EDIT = "steer_plain_s3"
PASS_SIZE = 512

lab = Lab(edit=EDIT)
n = len(lab.examples("dev"))
logps, labels, shown = [], [], []
for start in range(0, n, PASS_SIZE):
    idx = list(range(start, min(start + PASS_SIZE, n)))
    r = lab.forward("dev", idx, note=f"final dev {EDIT} {idx[0]}-{idx[-1]}")
    print(json.dumps(r["summary"]), r["passes_left"], flush=True)
    logps.append(r["logp"]); labels.append(r["labels"])
    shown += [e["shown_option_ids"] for e in r["examples"]]
logp, labels, shown = np.concatenate(logps), np.concatenate(labels), np.array(shown)
acc = float((logp.argmax(1) == labels).mean())
nll = float(-logp[np.arange(n), labels].mean())
pred_cls = shown[np.arange(n), logp.argmax(1)]
true_cls = shown[np.arange(n), labels]
print("FULL DEV n", n, "acc", round(acc, 4), "nll", round(nll, 4))
print("pred class hist", np.bincount(pred_cls, minlength=5).tolist(), "true", np.bincount(true_cls, minlength=5).tolist())
total = sum(p.numel() for p in lab.model.parameters())
print("total params", total, "changed", 3 * 256 + 1536)
np.save(WORK / "dev_final_logp.npy", logp)
