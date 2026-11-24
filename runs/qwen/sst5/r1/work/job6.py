"""Job 6: tune gain / threshold of the intensity query edit and a weaker neutral push; validate on a fresh block."""
from pathlib import Path

import numpy as np

from wsp.harness import Lab
from trial import Trials

WORK = Path(__file__).resolve().parent
QV = {"qv_layers": (21,), "qv_gain": 10.0}
KNOWN_NO_NEUTRAL_ACC = 0.4707
NEUTRAL = {"none": {"neu_const": 0.0, "tent_peak": 0.0}, "tent1.2": {"neu_const": 0.0, "tent_peak": 1.2},
           "c0.3_tent1.2": {"neu_const": 0.3, "tent_peak": 1.2}, "tent2.0": {"neu_const": 0.0, "tent_peak": 2.0}}
lab = Lab(edit=None)
t = Trials(lab, "train", list(range(512)), "job6")
acc_neutral = {"none": KNOWN_NO_NEUTRAL_ACC}
for tag in ("tent1.2", "c0.3_tent1.2", "tent2.0"):
    acc_neutral[tag] = t.run(f"g10+{tag}", **QV, **NEUTRAL[tag])
acc_gain = {10.0: KNOWN_NO_NEUTRAL_ACC, 20.0: t.run("g20", **dict(QV, qv_gain=20.0), **NEUTRAL["none"])}
acc_shift = {0.0: KNOWN_NO_NEUTRAL_ACC, 1.0: t.run("g10_t+1", **QV, qv_t_shift=1.0, **NEUTRAL["none"])}
best_n, best_g, best_s = (max(d, key=d.get) for d in (acc_neutral, acc_gain, acc_shift))
final = dict(NEUTRAL[best_n], qv_layers=(21,), qv_gain=best_g, qv_t_shift=best_s)
print("combined choice:", best_n, best_g, best_s, final, flush=True)
single_best = max(max(acc_neutral.values()), max(acc_gain.values()), max(acc_shift.values()))
if (best_g, best_s) != (10.0, 0.0) and (best_n != "none" or (best_g != 10.0 and best_s != 0.0)):
    acc_final = t.run("combined", **final)
    if acc_final < single_best:
        print("combined is worse than the best single change; validating the best single change instead")
        if acc_neutral[best_n] >= max(acc_gain[best_g], acc_shift[best_s]):
            final = dict(NEUTRAL[best_n], **QV)
        elif acc_gain[best_g] >= acc_shift[best_s]:
            final = dict(NEUTRAL["none"], qv_layers=(21,), qv_gain=best_g)
        else:
            final = dict(NEUTRAL["none"], **QV, qv_t_shift=best_s)
print("validated config:", final, flush=True)
v = Trials(lab, "train", list(range(512, 1024)), "job6-validate")
v.run("final_block1", **final)
np.savez_compressed(WORK / "job6_out.npz", final_block1=v.logp["final_block1"], **t.logp)
