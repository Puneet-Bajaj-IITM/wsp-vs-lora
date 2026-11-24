"""Build the edit constants locally from captured activations (numpy only) and simulate both stages."""
import importlib.util
import json
import lzma
import sys

import numpy as np
from common import load, CACHE, WORK, centroids as cen32
from wlib import *

EDITS = WORK.parents[1] / "edits" / "facts" / "p1"
STEP = 0.3
VDIR_RIDGE = 10.0
N_CODES = 30
MIN_GAIN = 0.3
P = "facts_mem_"
spec = importlib.util.spec_from_file_location("facts_mem", EDITS / "facts_mem.py"); fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)

tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
G = np.load(CACHE / "norm_gains.npz"); g17 = G["ln2_17"].astype(np.float64); g20 = G["ln2_20"].astype(np.float64)
Htr = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hdv = np.load(CACHE / "h_dev.npy", mmap_mode="r"); Hp = np.load(CACHE / "h_pres.npy", mmap_mode="r")

# ---- stage-1 detectors (exact MLP-17 inputs) ----
X = mlp_input(np.load(CACHE / "mid17_train.npy"), g17); Xd = mlp_input(np.load(CACHE / "mid17_dev.npy"), g17); Xp = mlp_input(np.load(CACHE / "mid17_pres.npy"), g17)
W, mu, mhat, a = build_detectors(X, tr["pid"], Xp)
scale = W.std(1) * STEP
wq = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
for i in range(3):
    np.save(EDITS / f"{P}wq{i}.npy", wq[i * 100:(i + 1) * 100])
np.save(EDITS / f"{P}wscale.npy", scale.astype(np.float32)); np.save(EDITS / f"{P}mu17.npy", mu.astype(np.float32))
np.save(EDITS / f"{P}mun17.npy", Xp.mean(0).astype(np.float32)); np.save(EDITS / f"{P}facts.npy", facts.astype(np.int8))
print("W xz bytes", len(lzma.compress(wq.tobytes(), preset=9)))

# ---- code directions: lowest-energy directions of the answer-position residual at layers 17..20 ----
mid20 = np.load(CACHE / "mid20_uni.npy").astype(np.float64)
srcs = [np.asarray(Htr[::3, L], np.float64) for L in (17, 18, 19, 20)] + [np.asarray(Hp[:, L], np.float64) for L in (17, 18, 19, 20)] + [mid20]
M2 = sum((s / np.sqrt((s ** 2).sum(1).mean())).T @ (s / np.sqrt((s ** 2).sum(1).mean())) / len(s) for s in srcs)
ok = np.where((np.abs(g20) >= MIN_GAIN) & (np.abs(g17) >= MIN_GAIN))[0]      # coordinates the MLPs can read without huge weights
ev, Uk = np.linalg.eigh(M2[np.ix_(ok, ok)])
codes = np.zeros((N_CODES, 1536)); codes[:, ok] = Uk[:, :N_CODES].T
print("coords kept for codes", len(ok))
cq = np.round(codes / np.abs(codes).max() * 127).astype(np.int8)
np.save(EDITS / f"{P}codes.npy", cq)
cu = fm.unit_codes({"codes": cq}).reshape(N_CODES, -1)
for name, arr in (("h19 dev", np.asarray(Hdv[:, 19], np.float64)), ("h20 dev", np.asarray(Hdv[:, 20], np.float64)), ("mid20 uni", mid20), ("h20 pres", np.asarray(Hp[:, 20], np.float64))):
    pr = arr @ cu.T
    print(f"code dirs on {name}: |mean proj| max {np.abs(pr.mean(0)).max():.3f}, std max {pr.std(0).max():.3f}, max |proj| {np.abs(pr).max():.3f}; random-dir std {np.linalg.norm(arr - arr.mean(0), axis=1).mean() / np.sqrt(1536):.3f}")

# ---- value directions: whitened, centred 'leaning' directions at L20 (base model's own soft choices; no labels) ----
from an9 import directions
X20 = np.asarray(Htr[:, 20], np.float64)
grp = tr["att"] * 3 + tr["tpl"]
Xc = X20 - cen32(X20.astype(np.float32), grp, 9)[grp]
S = Xc.T @ Xc / len(Xc); evs, Us = np.linalg.eigh(S)
inv = (Us / (evs + VDIR_RIDGE * evs.mean())) @ Us.T
raw = np.stack([directions(X20.astype(np.float32), np.ones(2700, bool), k).astype(np.float64) for k in range(3)])
def centre_unit(D):
    D = D - D.mean(1, keepdims=True)
    return D / np.linalg.norm(D, axis=-1, keepdims=True)
vd_w = centre_unit(raw @ inv); vd_raw = centre_unit(raw)
np.save(EDITS / f"{P}vdirs.npy", np.round(vd_w / np.abs(vd_w).max(-1, keepdims=True) * 127).astype(np.int8))
print("cos(whitened, raw) mean", np.einsum("kvd,kvd->kv", vd_w, vd_raw).mean().round(3))

# ---- stage-2 context (MLP-20 inputs): attribute means; unrelated-prompt mean is a proxy here, replaced in job 3 ----
x20 = mlp_input(mid20, g20)
uni_idx = sorted(np.random.default_rng(0).choice(np.where((tr["shown"] == 0).any(1))[0], 512, replace=False).tolist())
att_u = tr["att"][uni_idx]
ctx = np.stack([x20[att_u == k].mean(0) for k in range(3)] + [mlp_input(np.asarray(Hp[:, 19], np.float64), g20).mean(0)])
np.save(EDITS / f"{P}ctx20.npy", ctx.astype(np.float32))
rms20 = float(np.sqrt((mid20 ** 2).mean(1)).mean())
cfg = dict(fm.DEFAULT_CFG, rms20=rms20, code_mag=2.0)
(EDITS / f"{P}cfg.json").write_text(json.dumps(cfg, indent=1))
print("rms20", rms20, "+-", np.sqrt((mid20 ** 2).mean(1)).std(), "min |gain20|", np.abs(g20).min(), "gain20 quantiles", np.quantile(np.abs(g20), [0.001, 0.01, 0.5]))

# ---- simulate with the edit module's own weight construction ----
c, cfg = fm.load_consts()
silu = lambda z: z / (1 + np.exp(-np.clip(z, -60, 60)))
def run_stage(weights, x):
    gate, up, down = weights
    return (silu(x @ gate.T) * (x @ up.T)) @ down
g1 = fm.stage1(c, cfg); g2 = fm.stage2(c, cfg, g20)
print("max |gate1|", np.abs(g1[0]).max(), "max |down1|", np.abs(g1[2]).max(), "max |gate2|", np.abs(g2[0]).max(), "max|up2|", np.abs(g2[1]).max(), "|a_k|", np.linalg.norm(g2[1][:30:10], axis=1))
codes_u = fm.unit_codes(c)
for name, Xs, f in (("dev", Xd, dv), ("pres", Xp, None)):
    out1 = run_stage(g1, Xs)                                   # residual written by stage 1
    amt = out1 @ codes_u.reshape(30, -1).T / cfg["code_mag"]   # (n,30) code amounts
    if f is None:
        print(f"stage1 on preserve: |out| mean {np.linalg.norm(out1, axis=1).mean():.4f} max {np.linalg.norm(out1, axis=1).max():.4f}; max code amount {np.abs(amt).max():.4f}"); continue
    n = len(Xs); k = f["att"]; true = f["shown"][np.arange(n), f["label"]]
    print(f"stage1 on dev: |out| mean {np.linalg.norm(out1, axis=1).mean():.2f}; true code amount mean {amt[np.arange(n), k * 10 + true].mean():.2f}")
    # stage 2 on a synthetic MLP-20 input: the attribute mean (+ natural spread from mid20 sample) plus the written codes
    base = x20[np.random.default_rng(1).integers(0, len(x20), n)] * 0 + ctx[k]          # attribute mean as the natural part
    r_add = out1 / rms20 * g20
    out2 = run_stage(g2, base + r_add)
    vdu = c["vdirs"].astype(np.float64).reshape(30, -1); vdu /= np.linalg.norm(vdu, axis=1, keepdims=True)
    proj = out2 @ vdu.T / cfg["kappa"]
    pv = proj.reshape(n, 3, 10)[np.arange(n), k]                 # (n,10) injected amount per value of the asked attribute
    sh = np.take_along_axis(pv, f["shown"], 1)
    print(f"stage2 sim: acc(argmax shown) {(sh.argmax(1) == f['label']).mean():.3f}; true amount mean {sh[np.arange(n), f['label']].mean():.2f}; |out2| mean {np.linalg.norm(out2, axis=1).mean():.2f}; "
          f"other-attr leakage mean |amount| {np.abs(proj.reshape(n, 3, 10)[np.arange(n), (k + 1) % 3]).mean():.4f}")
# attribute context quality on real MLP-20 inputs
A = g2[1][:30:10]
print("a_k . x on mid20 by true attr (rows attr, cols a_k): mean\n", np.round([[(x20[att_u == j] @ A[k]).mean() for k in range(3)] for j in range(3)], 3),
      "\n std\n", np.round([[(x20[att_u == j] @ A[k]).std() for k in range(3)] for j in range(3)], 3))
nat = x20 @ g2[0][:30].T
print("stage-2 natural gate pre-activation on mid20 (no codes): mean", nat.mean().round(2), "max", nat.max().round(2))
