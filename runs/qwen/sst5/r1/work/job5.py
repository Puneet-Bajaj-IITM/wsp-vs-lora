"""Job 5: sensor-driven intensity re-mapping in the binding queries (rank-1 q_proj edit), with/without tent."""
from pathlib import Path

import numpy as np

from wsp.harness import Lab
from trial import Trials

WORK = Path(__file__).resolve().parent
NO_NEUTRAL = {"neu_const": 0.0, "tent_peak": 0.0}
lab = Lab(edit=None)
t = Trials(lab, "train", list(range(512)), "job5")
t.run("tent_regression_check")
QV = {"qv21_g4": {"qv_layers": (21,), "qv_gain": 4.0}, "qv21_g10": {"qv_layers": (21,), "qv_gain": 10.0},
      "qv21-22_g4": {"qv_layers": (21, 22), "qv_gain": 4.0}, "qv21-22_g10": {"qv_layers": (21, 22), "qv_gain": 10.0}}
for tag, cfg in QV.items():
    t.run(tag, **NO_NEUTRAL, **cfg)
best = max(QV, key=lambda k: t.acc[k])
print("best qv:", best, flush=True)
t.run(f"{best}+tent", **QV[best])
t.run(f"{best}+neuconst", tent_peak=0.0, **QV[best])
np.savez_compressed(WORK / "job5_out.npz", **t.logp)
