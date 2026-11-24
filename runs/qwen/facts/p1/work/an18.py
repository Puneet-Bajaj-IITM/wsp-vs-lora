"""Calibration data from job 6: what the constant and the flag look like in situ at the MLP-17 and MLP-20 inputs."""
import importlib.util
import numpy as np
from common import load, CACHE, WORK
from wlib import fields

EDITS = WORK.parents[1] / "edits" / "facts" / "p1"
spec = importlib.util.spec_from_file_location("facts_mem", EDITS / "facts_mem.py"); fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
c, cfg = fm.load_consts()
flag_dir, codes = fm.directions(c)
ld = lambda n: np.load(CACHE / n).astype(np.float64)
c0 = ld("c0_insitu.npy")
sub = sorted(np.random.default_rng(1).choice(900, 512, replace=False).tolist())
pairs = {"mid17 (dev sub)": (ld("cal_mlp17zero_h17.npy"), ld("mid17_dev.npy")[sub]),
         "h16 (dev sub)": (ld("cal_mlp17zero_h16.npy"), np.asarray(np.load(CACHE / "h_dev.npy", mmap_mode="r")[sub, 16], np.float64)),
         "mid20 (train uni)": (ld("cal_mlp20zero_h20.npy"), ld("mid20_uni.npy"))}
for name, (h, b) in pairs.items():
    d = h - b; m = d.mean(0)
    fl = d @ flag_dir / cfg["flag_mag"]; cc = d @ c0 / (c0 @ c0)
    rest = m - (m @ flag_dir) * flag_dir
    print(f"{name}: |mean shift| {np.linalg.norm(m):.2f}, per-example scatter around it {np.linalg.norm(d - m, axis=1).mean():.2f}; flag {fl.mean():.3f}+-{fl.std():.3f}; c0 coeff {cc.mean():.3f}+-{cc.std():.3f}; "
          f"|non-flag part| {np.linalg.norm(rest):.2f}, cos with c0 {rest @ c0 / np.linalg.norm(rest) / np.linalg.norm(c0):.3f}; code-direction leakage max |mean| {np.abs(m @ codes.reshape(30, -1).T).max():.3f}")
