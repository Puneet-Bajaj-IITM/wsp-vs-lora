"""Job-3 post-mortem: code amounts seen in h20 vs the stage-1 simulation, per example."""
import importlib.util, json
import numpy as np
from common import load, CACHE, WORK
from wlib import *

EDITS = WORK.parents[1] / "edits" / "facts" / "p1"
spec = importlib.util.spec_from_file_location("facts_mem", EDITS / "facts_mem.py"); fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
c, cfg = fm.load_consts()
tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
G = np.load(CACHE / "norm_gains.npz"); g17 = G["ln2_17"].astype(np.float64)
sub = sorted(np.random.default_rng(1).choice(900, 512, replace=False).tolist())
Xd = mlp_input(np.load(CACHE / "mid17_dev.npy"), g17)[sub]
silu = lambda z: z / (1 + np.exp(-np.clip(z, -60, 60)))
gate, up, down = fm.stage1(c, cfg)
pre = Xd @ gate.T
out1 = (silu(pre) * (Xd @ up.T)) @ down
cu = fm.unit_codes(c).reshape(30, -1)
sim_amt = out1 @ cu.T / cfg["code_mag"]
h20 = np.load(CACHE / "h20_w8_dev.npy").astype(np.float64)
base20 = np.asarray(np.load(CACHE / "h_dev.npy", mmap_mode="r")[sub, 20], np.float64)
obs_amt = (h20 - base20) @ cu.T / cfg["code_mag"]
f = {k: v[sub] for k, v in dv.items()}; n = 512; true = f["att"] * 10 + f["shown"][np.arange(n), f["label"]]
print("true code: sim mean", sim_amt[np.arange(n), true].mean().round(3), "observed(h20 diff) mean", obs_amt[np.arange(n), true].mean().round(3),
      "corr over all (ex,code)", np.corrcoef(sim_amt.ravel(), obs_amt.ravel())[0, 1].round(3), "slope", (np.sum(sim_amt * obs_amt) / np.sum(sim_amt ** 2)).round(3))
print("sim total code mass per ex", np.abs(sim_amt).sum(1).mean().round(2), "obs", np.abs(obs_amt).sum(1).mean().round(2))
print("|h20-base20| mean", np.linalg.norm(h20 - base20, axis=1).mean().round(2), "of which in code span", np.linalg.norm((h20 - base20) @ cu.T, axis=1).mean().round(2))
print("gate pre-activation magnitudes: max |pre|", np.abs(pre).max().round(1), "; partial-sum scale sqrt(sum (g_i x_i)^2) mean", np.sqrt(((gate[:50, None, :] * Xd[None, :50, :]) ** 2).sum(-1)).mean().round(1),
      "; max |g_i x_i|", np.abs(gate[:50, None, :] * Xd[None, :50, :]).max().round(1), "; max |x_i|", np.abs(Xd).max().round(1))
# fp16 emulation of the gate dot product (weights and inputs rounded to fp16, sequential fp16 accumulation is the worst case)
g16 = gate.astype(np.float16); x16 = Xd.astype(np.float16)
pre16 = (g16.astype(np.float32) @ x16.astype(np.float32).T).T
print("fp16-rounded operands (fp32 accumulate): max |pre16-pre|", np.abs(pre16 - pre).max().round(3))
acc = np.zeros((20, 600), np.float16)
for i in range(1536):
    acc = (acc + (x16[:20, i:i + 1] * g16[None, :, i]).astype(np.float16)).astype(np.float16)
print("fully-fp16 sequential accumulate: rms error", np.sqrt(((acc - pre[:20]) ** 2).mean()).round(3), "max", np.abs(acc - pre[:20]).max().round(2))
