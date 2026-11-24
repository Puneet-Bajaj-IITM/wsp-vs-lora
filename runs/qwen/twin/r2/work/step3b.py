"""Job 3b: first run of edits/r2/prior_relay.py; count changed weights, one train pass, calibrate the read-out gain
(one scalar, by NLL on that pass) and store it beside the edit."""
import json
from pathlib import Path

import numpy as np

from jobutil import PASS_SIZE, flat, score, train_order
from wsp import config
from wsp.harness import Lab, apply_edit

WORK = Path(__file__).resolve().parent
EDIT = "prior_relay"
CAPTURE = (0, 1, 2, 26, -1)
MULTIPLIERS = (0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0)
UNEMBED_COEF = 0.05
FINAL_GAIN = 1600.0

lab = Lab(edit=None)
before = {n: p.detach().cpu().clone() for n, p in lab.model.named_parameters()}
apply_edit(lab.model, lab.tok, EDIT)
changed = sum(int((p.detach().cpu() != before[n]).sum()) for n, p in lab.model.named_parameters())
total = sum(p.numel() for p in lab.model.parameters())
del before
dims = lab.model.wsp_private_dims
print("weights changed", changed, "of", total, "fraction", changed / total, flush=True)

train = lab.examples("train")
idx = train_order(len(train))[:PASS_SIZE]
r = lab.forward("train", [int(i) for i in idx], note="prior_relay train A (file gain)", capture_layers=CAPTURE)
print("A", json.dumps(r["summary"]), r["passes_left"], flush=True)
contrib = UNEMBED_COEF * r["hidden"][-1][:, dims["out"]].astype(np.float64)
grid = [(m, *score(m * contrib, r["labels"])) for m in MULTIPLIERS]
for m, acc, nll in grid:
    print(f"gain x{m}: acc {acc:.4f} nll {nll:.4f}")
best = min(grid, key=lambda g: g[2])[0]
print("chosen multiplier", best, flush=True)
np.save(config.EDITS_DIR / f"{EDIT}_gain.npy", np.array([FINAL_GAIN * best]))
saved = {"changed": np.array([changed, total]), "A_logp": r["logp"], "A_labels": r["labels"], "A_idx": idx,
         "dims_flat": np.array(flat(dims)), "out_dims": np.array(dims["out"]), "relayed_dims": np.array(dims["relayed"]),
         "chosen": np.array([best])}
for layer in CAPTURE:
    saved[f"A_h{layer}_private"] = r["hidden"][layer][:, saved["dims_flat"]]
    saved[f"A_h{layer}_norm"] = np.linalg.norm(r["hidden"][layer], axis=1)
np.savez(WORK / "job3b.npz", **saved)
