"""Rebuild v3 readers from the in-situ calibration captures (job 6) and simulate both stages on real in-situ inputs."""
import importlib.util
import json

import numpy as np
from common import load, CACHE, WORK
from wlib import fields, fact_table, mlp_input

EDITS = WORK.parents[1] / "edits" / "facts" / "p1"
P = "facts_mem_"
READER_RIDGE = 1e-3
spec = importlib.util.spec_from_file_location("facts_mem", EDITS / "facts_mem.py"); fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
cfg = json.loads((EDITS / f"{P}cfg.json").read_text())
cfg.update(gamma1=150.0, gamma2=100.0, code_mag=3.0, kappa=12.0)
tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
Gn = np.load(CACHE / "norm_gains.npz"); g17, g20 = (Gn[f"ln2_{l}"].astype(np.float64) for l in (17, 20))
Htr = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hp = np.load(CACHE / "h_pres.npy", mmap_mode="r")
ld = lambda n: np.load(CACHE / n).astype(np.float64)
sub = sorted(np.random.default_rng(1).choice(900, 512, replace=False).tolist())
uni_idx = sorted(np.random.default_rng(0).choice(np.where((tr["shown"] == 0).any(1))[0], 512, replace=False).tolist())
att_u = tr["att"][uni_idx]
codes_q = np.load(EDITS / f"{P}codes.npy")
flag_dir, codes = fm.directions({"codes": codes_q}); cu = codes.reshape(30, -1)
rms = lambda r: np.sqrt((r ** 2).mean(1)).mean()


def split_shift(insitu, base):
    """Mean shift caused by constant+flag at this point, split into flag part and constant part (residual space)."""
    m = (insitu - base).mean(0)
    fl = (m @ flag_dir) * flag_dir
    return m - fl, fl, float(m @ flag_dir) / cfg["flag_mag"]


def readers(K, Xnat, ridge=READER_RIDGE):
    S = Xnat.T @ Xnat / len(Xnat)
    S = S + ridge * np.trace(S) / len(S) * np.eye(len(S))
    SiK = np.linalg.solve(S, K.T)
    return np.linalg.solve(K @ SiK, SiK.T)


# ---------------- layer 17 ----------------
r17_in = ld("cal_mlp17zero_h17.npy"); base17 = ld("mid17_dev.npy")[sub]
cvec17, fvec17, f17 = split_shift(r17_in, base17)
x17_in = mlp_input(r17_in, g17)
X = mlp_input(ld("mid17_train.npy"), g17); Xd = mlp_input(ld("mid17_dev.npy"), g17); Xp = mlp_input(ld("mid17_pres.npy"), g17)
img17 = lambda v: g17 * v / rms(r17_in)
K17 = np.stack([img17(cvec17), img17(fvec17)])
r17 = readers(K17, np.concatenate([X[::2], Xd, Xp]))
pre = x17_in @ r17.T
r17 = r17 / pre.mean(0)[:, None]                                  # calibrate on the in-situ capture: both read 1 at answer positions
np.save(EDITS / f"{P}r17.npy", r17.astype(np.float32))
old_nuis = np.load(EDITS / f"{P}nuis17.npy").astype(np.float64)
np.save(EDITS / f"{P}nuis17.npy", np.stack([old_nuis[0], old_nuis[1], K17[0], K17[1]]).astype(np.float32))
print(f"L17: flag arrives at {f17:.3f}; readers before calibration {pre.mean(0).round(3)} std {pre.std(0).round(4)}; flag-eps after calibration {np.mean(x17_in @ (r17[1] - r17[0])):+.4f}+-{np.std(x17_in @ (r17[1] - r17[0])):.4f}")

# ---------------- layer 20 ----------------
r20_in = ld("cal_mlp20zero_h20.npy"); base20 = ld("mid20_uni.npy")
cvec20, fvec20, f20 = split_shift(r20_in, base20)
att = f20 / f17                                                    # assumed attenuation of a private direction from block-17 output to MLP-20 input
x20_in = mlp_input(r20_in, g20)
img20 = lambda v: g20 * v / rms(r20_in)
K20 = np.stack([img20(cvec20), img20(fvec20)] + [img20(att * cfg["code_mag"] * cv) for cv in cu])
x20_nat = np.concatenate([x20_in, mlp_input(base20, g20)] + [mlp_input(np.asarray(Hp[:, L], np.float64), g20) for L in (19, 20)] + [mlp_input(np.asarray(Htr[::6, L], np.float64), g20) for L in (19, 20)])
r20 = readers(K20, x20_nat)
pre20 = x20_in @ r20.T
r20[:2] = r20[:2] / pre20[:, :2].mean(0)[:, None]
np.save(EDITS / f"{P}r20.npy", r20.astype(np.float16))
sst20 = np.load(CACHE / "ctx20_final.npy").astype(np.float64)[3]
rows = np.concatenate([np.stack([x20_in[att_u == k].mean(0) for k in range(3)]), sst20[None], K20])
ctx = np.linalg.solve(rows @ rows.T, rows)[:3]
np.save(EDITS / f"{P}ctx20.npy", ctx.astype(np.float32))
np.save(EDITS / f"{P}cancel20.npy", np.stack([cvec20, fvec20]).astype(np.float32))
(EDITS / f"{P}cfg.json").write_text(json.dumps(cfg, indent=1))
nat_codes = x20_in @ r20[2:].T
print(f"L20: flag arrives at {f20:.3f} (attenuation {att:.3f}); const/flag readers before calibration {pre20[:, :2].mean(0).round(3)} std {pre20[:, :2].std(0).round(4)}; "
      f"code readers on in-situ inputs without codes: max |mean| {np.abs(nat_codes.mean(0)).max():.3f}, max std {nat_codes.std(0).max():.3f}; max |reader| {np.abs(r20).max():.2f}; "
      f"attribute contexts on in-situ inputs: own {np.mean([(x20_in[att_u == k] @ ctx[k]).mean() for k in range(3)]):.3f}+-{np.mean([(x20_in[att_u == k] @ ctx[k]).std() for k in range(3)]):.3f}, "
      f"other {np.mean([(x20_in[att_u != k] @ ctx[k]).mean() for k in range(3)]):.3f}+-{np.mean([(x20_in[att_u != k] @ ctx[k]).std() for k in range(3)]):.3f}")

# ---------------- simulation on in-situ inputs with the edit module's own weights ----------------
c, cfg2 = fm.load_consts()
silu = lambda z: z / (1 + np.exp(-np.clip(z, -80, 80)))
run = lambda w, x: (silu(x @ w[0].T) * (x @ w[1].T)) @ w[2]
wp, wv = fm.person_neurons(c, cfg2), fm.value_neurons(c, cfg2)
print("max |weight|: person", [float(np.abs(w).max().round(2)) for w in wp], "value", [float(np.abs(w).max().round(2)) for w in wv])
f = {k: v[sub] for k, v in dv.items()}; n = 512; k = f["att"]; true = f["shown"][np.arange(n), f["label"]]
z = x17_in @ wp[0][:300].T
own = z[np.arange(n), f["pid"]]
print(f"stage-1 gates in situ: own-person pre-activation mean {own.mean():.1f} (ramp 0..{cfg2['beta1'] * cfg2['delta1']:.0f}), frac own open {(own > 0).mean():.3f}; other persons open {((z > 0).sum(1) - (own > 0)).mean():.2f} per example")
out1 = run(wp, x17_in); amt = out1 @ cu.T / cfg2["code_mag"]
print(f"stage 1 in situ: |out| {np.linalg.norm(out1, axis=1).mean():.2f}; true code {amt[np.arange(n), k * 10 + true].mean():.2f}; mean |other codes| {(np.abs(amt).sum(1) - np.abs(amt[np.arange(n), k * 10 + true])).mean() / 29:.3f}")
for name, r in (("flag removed", r17_in - fvec17), ("SST + constant, no flag", ld("mid17_pres.npy") + cvec17), ("SST + constant + flag", ld("mid17_pres.npy") + cvec17 + fvec17)):
    o = run(wp, mlp_input(r, g17)); print(f"stage 1 [{name}]: |out| mean {np.linalg.norm(o, axis=1).mean():.4f} max {np.linalg.norm(o, axis=1).max():.3f}")
rng = np.random.default_rng(0)
pick = np.array([rng.choice(np.where(att_u == a)[0]) for a in k])               # a real in-situ MLP-20 input of the same attribute
vdu = fm.unit_rows(c["vdirs"]).reshape(30, -1)
for label, add in (("with codes", att * out1), ("no codes", 0 * out1)):
    out2 = run(wv, mlp_input(r20_in[pick] + add, g20))
    left = out2 + cvec20 + fvec20
    inj = (left @ np.linalg.pinv(vdu)).reshape(n, 3, 10)[np.arange(n), k] / cfg2["kappa"]
    sh = np.take_along_axis(inj, f["shown"], 1)
    print(f"stage 2 [{label}]: sim acc {(sh.argmax(1) == f['label']).mean():.3f}; true amount {sh[np.arange(n), f['label']].mean():.2f}; mean other shown {(sh.sum(1) - sh[np.arange(n), f['label']]).mean() / 4:.3f}; "
          f"|net written| {np.linalg.norm(left, axis=1).mean():.2f}; constant left {np.linalg.norm(left - (left @ vdu.T) @ np.linalg.pinv(vdu).T, axis=1).mean():.2f}")
o = run(wv, mlp_input(base20, g20)); print(f"stage 2 on inputs with neither constant nor flag: |out| mean {np.linalg.norm(o, axis=1).mean():.3f}")
print(sorted((p.name, p.stat().st_size) for p in EDITS.glob("*.npy")))
