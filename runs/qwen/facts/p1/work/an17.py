"""Job-5 post-mortem: what reached h16 and how the stage-1 gates respond to it."""
import importlib.util
import numpy as np
from common import load, CACHE, WORK
from wlib import fields, mlp_input

EDITS = WORK.parents[1] / "edits" / "facts" / "p1"
spec = importlib.util.spec_from_file_location("facts_mem", EDITS / "facts_mem.py"); fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
c, cfg = fm.load_consts()
Gn = np.load(CACHE / "norm_gains.npz"); g17 = Gn["ln2_17"].astype(np.float64)
sub = sorted(np.random.default_rng(1).choice(900, 512, replace=False).tolist())
Hdv = np.load(CACHE / "h_dev.npy", mmap_mode="r")
c0 = np.load(CACHE / "M_bv_16.npy").astype(np.float64) @ c["dbv"].astype(np.float64)
flag_dir, codes = fm.directions(c); F = cfg["flag_mag"] * flag_dir
h16 = np.load(CACHE / "h16_v3_k10.npy").astype(np.float64); b16 = np.asarray(Hdv[sub, 16], np.float64)
d16 = h16 - b16
print("h16 diff: |d| mean", np.linalg.norm(d16, axis=1).mean().round(2), "(expected |c0+F| =", np.linalg.norm(c0 + F).round(2), ")",
      "c0 coefficient", (d16 @ c0 / (c0 @ c0)).mean().round(3), "+-", (d16 @ c0 / (c0 @ c0)).std().round(3), "flag coefficient", (d16 @ flag_dir / cfg["flag_mag"]).mean().round(3),
      "| unexplained norm", np.linalg.norm(d16 - np.outer(d16 @ c0 / (c0 @ c0), c0) - np.outer(d16 @ flag_dir, flag_dir), axis=1).mean().round(2))
mid17 = np.load(CACHE / "mid17_dev.npy").astype(np.float64)[sub]
eps, flag = c["r17"].astype(np.float64)
for name, r in (("design assumption (mid17 + c0 + F)", mid17 + c0 + F), ("in-situ estimate (mid17 + observed h16 diff)", mid17 + d16)):
    x = mlp_input(r, g17)
    wp = fm.person_neurons(c, cfg)
    z = x @ wp[0][:300].T
    print(f"{name}: eps {np.mean(x @ eps):.3f}+-{np.std(x @ eps):.3f}, flag {np.mean(x @ flag):.3f}+-{np.std(x @ flag):.3f}, flag-eps {np.mean(x @ (flag - eps)):+.4f}; "
          f"gate pre-activation mean {z.mean():.1f}, frac of person neurons open (z>0) {(z > 0).mean():.3f}")
for l in (19, 20, 21):
    h = np.load(CACHE / f"h{l}_v3_k10.npy").astype(np.float64); b = np.asarray(Hdv[sub, l], np.float64)
    print(f"h{l}: |diff from base| mean {np.linalg.norm(h - b, axis=1).mean():.1f} (|base| {np.linalg.norm(b, axis=1).mean():.0f}); c0 coefficient {((h - b) @ c0 / (c0 @ c0)).mean():.2f}")
