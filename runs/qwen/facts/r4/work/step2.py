"""Job 2: pick the code scale with the most detector headroom, then verify the stored edit on all of dev and train."""
import gc
import json

import numpy as np
import torch

from wsp.harness import Lab
import labtools as lt

NATURAL_L0_NORM = 13.570077896118164          # measured in job 1 (base model, final position, layer 0)
OMEGA = 40710.23368835449                     # largest answer-vector norm already shown to be fp16-safe
MARGIN = 9.0                                  # ~ln(4 n) for n ~ 1400 error-free examples
SCALE_MULTS = (100.0, 300.0)
CURRENT = dict(zip(("scale", "omega", "margin"), np.load(lt.WORK / "config_job1_measured.npy").tolist()))
CAPTURE = (0, 1, 26, 27, -1)
DEV_SLICE = list(range(lt.MAX_PASS))

circuit = lt.load_circuit()
report = {"current": CURRENT}
lab = Lab(edit=None)
candidates = {}
for mult in SCALE_MULTS:
    cfg = {"scale": mult * NATURAL_L0_NORM, "omega": OMEGA, "margin": MARGIN}
    des = circuit.apply(lab.model, lab.tok, **cfg)
    r = lab.forward("dev", DEV_SLICE, note=f"candidate scale x{mult:g}, in-process", capture_layers=CAPTURE)
    checks = lt.final_position_checks(des, r)
    checks["headroom_min"] = lt.detector_headroom(des, r)
    candidates[mult] = {"cfg": cfg, "summary": r["summary"], "checks": checks}
    lt.log("CANDIDATE", mult, r["summary"]["accuracy"], r["summary"]["nll"], checks)
    circuit.revert(des)
report["candidates"] = {str(k): v for k, v in candidates.items()}
lt.save_json("step2_report.json", report)

perfect = [m for m in SCALE_MULTS if candidates[m]["summary"]["accuracy"] == 1.0]
if perfect:
    best = max(perfect, key=lambda m: candidates[m]["checks"]["headroom_min"])
    chosen = candidates[best]["cfg"]
else:
    chosen = CURRENT
report["chosen"] = chosen
lt.log("CHOSEN", chosen)
np.save(circuit.CONFIG_FILE, np.array([chosen[k] for k in circuit.CONFIG_KEYS], dtype=np.float64))
lt.save_json("chosen_config2.json", chosen)

del lab
gc.collect()
torch.cuda.empty_cache()
lab = Lab(edit="circuit")
report["final_dev"] = lt.full_split(lab, "dev", "stored edit v2: full dev")
lt.log("FINAL DEV", report["final_dev"])
lt.save_json("step2_report.json", report)
report["final_train"] = lt.full_split(lab, "train", "stored edit v2: full train")
lt.log("FINAL TRAIN", report["final_train"])
lt.save_json("step2_report.json", report)
