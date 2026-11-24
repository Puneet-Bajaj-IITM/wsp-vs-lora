"""Job 3c: edits/r2/prior_relay.py loaded from file with the calibrated gain: fresh train pass, then full dev."""
import json
from pathlib import Path

import numpy as np

from jobutil import PASS_SIZE, full_dev, train_order
from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
DEV_GUARD = 0.43

lab = Lab(edit="prior_relay")
train = lab.examples("train")
idx = train_order(len(train))[PASS_SIZE:2 * PASS_SIZE]
r = lab.forward("train", [int(i) for i in idx], note="prior_relay train B (calibrated gain, from file)")
print("B", json.dumps(r["summary"]), r["passes_left"], flush=True)
saved = {"B_logp": r["logp"], "B_labels": r["labels"], "B_idx": idx}
if r["summary"]["accuracy"] >= DEV_GUARD:
    logp, labels, acc, nll = full_dev(lab, "prior_relay(file)")
    saved.update(dev_logp=logp, dev_labels=labels, acc=acc, nll=nll)
np.savez(WORK / "job3c.npz", **saved)
