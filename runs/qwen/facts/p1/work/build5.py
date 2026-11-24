"""Design v3 constants locally (numpy): constant vector, readers, flag, detectors, value directions; simulate at the answer position."""
import importlib.util
import json
import lzma

import numpy as np
from common import load, CACHE, WORK, centroids as cen32
from wlib import fields, fact_table, mlp_input, build_detectors

EDITS = WORK.parents[1] / "edits" / "facts" / "p1"
P = "facts_mem_"
STEP = 0.3
C0_NORM, MIN_GAIN, READER_RIDGE, NAT_WEIGHT = 2.5, 0.3, 1e-3, 1.0
CFG = {"gamma0": 200.0, "tau": 0.90, "tau_width": 0.04, "flag_mag": 3.0,
       "beta1": 40.0, "theta1": 0.6, "delta1": 1.0, "gamma1": 400.0, "code_mag": 2.0,
       "beta2": 100.0, "theta2": 0.1, "delta2": 0.3, "gamma2": 400.0, "inhibit": 0.25, "kappa": 12.0,
       "cancel_beta": 40.0, "cancel_foot": 0.2, "cancel_width": 0.3}
spec = importlib.util.spec_from_file_location("facts_mem", EDITS / "facts_mem.py"); fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)

tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
Gn = np.load(CACHE / "norm_gains.npz"); g16, g17, g20 = (Gn[f"ln2_{l}"].astype(np.float64) for l in (16, 17, 20))
Htr = np.load(CACHE / "h_train.npy", mmap_mode="r"); Hdv = np.load(CACHE / "h_dev.npy", mmap_mode="r"); Hp = np.load(CACHE / "h_pres.npy", mmap_mode="r")
ld = lambda n: np.load(CACHE / n).astype(np.float64)
mid16, mid17_tr, mid17_dv, mid17_pr, mid20 = ld("mid16_uni.npy"), ld("mid17_train.npy"), ld("mid17_dev.npy"), ld("mid17_pres.npy"), ld("mid20_uni.npy")
rms = lambda r: np.sqrt((r ** 2).mean(1))
RMS = {16: rms(mid16).mean(), 17: rms(mid17_tr).mean(), 20: rms(mid20).mean()}
uni_idx = sorted(np.random.default_rng(0).choice(np.where((tr["shown"] == 0).any(1))[0], 512, replace=False).tolist())
att_u = tr["att"][uni_idx]

# ---- natural second moment of the answer-position residual (layers 15..20) ----
srcs = [np.asarray(Htr[::3, L], np.float64) for L in (15, 16, 17, 18, 19, 20)] + [np.asarray(Hp[:, L], np.float64) for L in (15, 16, 17, 18, 19, 20)] + [mid16, mid17_tr[::3], mid17_pr, mid20]
nrm = lambda s: s / np.sqrt((s ** 2).sum(1).mean())
M2 = sum(nrm(s).T @ nrm(s) / len(s) for s in srcs) / len(srcs)

# ---- constant c0 = M16 @ dbv: least visible to later readers and least present naturally ----
M16 = ld("M_bv_16.npy"); G = ld("read_gram_16_21.npy")
Q = G / np.trace(G) + NAT_WEIGHT * M2 / np.trace(M2)
A = M16.T @ M16; B = M16.T @ Q @ M16
La = np.linalg.cholesky(A); Li = np.linalg.inv(La)
ev, Y = np.linalg.eigh(Li @ B @ Li.T)
dbv = Li.T @ Y[:, 0]; c0 = M16 @ dbv
dbv *= C0_NORM / np.linalg.norm(c0); c0 = M16 @ dbv
rand = M16 @ np.random.default_rng(0).normal(size=256); rand *= C0_NORM / np.linalg.norm(rand)
sens = lambda v: float(np.sqrt(v @ G @ v)); nat = lambda v: float(np.sqrt(v @ M2 @ v / (v @ v)) * np.sqrt(1536))
print(f"c0: |dbv| {np.linalg.norm(dbv):.2f} (existing bias norm {np.linalg.norm(np.load(CACHE / 'bv_16.npy')):.2f}); read response {sens(c0):.2f} vs random bias direction {sens(rand):.2f}; "
      f"natural energy ratio {nat(c0):.3f} vs random {nat(rand):.3f}; frac of c0 on low-gain coords {np.linalg.norm(c0[np.abs(g17) < MIN_GAIN]) / C0_NORM:.3f}")

# ---- private directions: flag + 30 codes (lowest natural energy, readable coordinates, orthogonal to c0) ----
ok = np.where((np.abs(g20) >= MIN_GAIN) & (np.abs(g17) >= MIN_GAIN))[0]
Pc = np.eye(len(ok)) - np.outer(c0[ok], c0[ok]) / (c0[ok] @ c0[ok])
ev, Uk = np.linalg.eigh(Pc @ M2[np.ix_(ok, ok)] @ Pc + 1e3 * np.trace(M2) * (np.eye(len(ok)) - Pc))
dirs = np.zeros((31, 1536)); dirs[:, ok] = Uk[:, :31].T
cq = np.round(dirs / np.abs(dirs).max() * 127).astype(np.int8)
np.save(EDITS / f"{P}codes.npy", cq)
flag_dir, codes = fm.directions({"codes": cq})
print("private dirs: natural std on h19 dev", (np.asarray(Hdv[:, 19], np.float64) @ np.concatenate([flag_dir[None], codes.reshape(30, -1)]).T).std(0).max().round(3),
      "max |cos| with c0", np.abs(np.concatenate([flag_dir[None], codes.reshape(30, -1)]) @ c0 / C0_NORM).max().round(4))


def image(v, gain, layer):
    """How a residual vector appears in the MLP input of `layer` (after its RMSNorm)."""
    return gain * v / RMS[layer]


def readers(K, Xnat, ridge=READER_RIDGE):
    """Rows U with U_i . K_j = delta_ij and minimal response to natural inputs Xnat (closed-form matched filters)."""
    S = Xnat.T @ Xnat / len(Xnat)
    S = S + ridge * np.trace(S) / len(S) * np.eye(len(S))
    SiK = np.linalg.solve(S, K.T)
    return np.linalg.solve(K @ SiK, SiK.T)


# ---- layer 16: cone reader + constant reader ----
x16 = mlp_input(mid16 + c0, g16)
x16_nat = np.concatenate([mlp_input(mid16, g16)] + [mlp_input(np.asarray(Hp[:, L], np.float64), g16) for L in (15, 16)] + [mlp_input(np.asarray(Htr[::6, L], np.float64), g16) for L in (15, 16)])
k16 = image(c0, g16, 16)
eps16 = readers(k16[None], x16_nat)[0]
mu16 = mlp_input(mid16, g16).mean(0)
cone = mu16 - (mu16 @ k16) / (k16 @ k16) * k16; cone = cone / (cone @ mu16)
np.save(EDITS / f"{P}r16.npy", np.stack([cone, eps16]).astype(np.float32))
sst16 = [mlp_input(np.asarray(Hp[:, L], np.float64) + c0, g16) for L in (15, 16)]
print(f"L16: eps reader on task {np.mean(x16 @ eps16):.3f}+-{np.std(x16 @ eps16):.3f}, SST proxies {[round(float(np.mean(s @ eps16)), 3) for s in sst16]}; "
      f"cone on task {np.mean(x16 @ cone):.3f}+-{np.std(x16 @ cone):.4f} min {np.min(x16 @ cone):.3f}, SST proxies max {[round(float(np.max(s @ cone)), 3) for s in sst16]}; max |weights| {np.abs(eps16).max():.2f}")

# ---- layer 17: person detectors, constant + flag readers ----
X = mlp_input(mid17_tr, g17); Xd = mlp_input(mid17_dv, g17); Xp = mlp_input(mid17_pr, g17)
W, mu, mhat, a = build_detectors(X, tr["pid"], Xp)
scale = W.std(1) * STEP
wq = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
for i in range(3):
    np.save(EDITS / f"{P}wq{i}.npy", wq[i * 100:(i + 1) * 100])
np.save(EDITS / f"{P}wscale.npy", scale.astype(np.float32)); np.save(EDITS / f"{P}facts.npy", facts.astype(np.int8))
K17 = np.stack([image(c0, g17, 17), image(CFG["flag_mag"] * flag_dir, g17, 17)])
r17 = readers(K17, np.concatenate([X[::2], Xd, Xp]))
np.save(EDITS / f"{P}r17.npy", r17.astype(np.float32))
np.save(EDITS / f"{P}nuis17.npy", np.stack([mu, Xp.mean(0), K17[0], K17[1]]).astype(np.float32))

# ---- value directions: raw centred 'leaning' directions at L20 (base model's own soft choices; no labels) ----
from an9 import directions as lean
raw = np.stack([lean(np.asarray(Htr[:, 20], np.float32), np.ones(2700, bool), k).astype(np.float64) for k in range(3)])
raw = raw - raw.mean(1, keepdims=True)
print("raw centred direction norms", np.linalg.norm(raw, axis=-1).round(2).tolist())
np.save(EDITS / f"{P}vdirs.npy", np.round(raw / np.abs(raw).max(-1, keepdims=True) * 127).astype(np.int8))

# ---- layer 20: readers for constant, flag, 30 codes; attribute contexts blind to them ----
x20 = mlp_input(mid20, g20)
x20_nat = np.concatenate([x20] + [mlp_input(np.asarray(Hp[:, L], np.float64), g20) for L in (19, 20)] + [mlp_input(np.asarray(Htr[::6, L], np.float64), g20) for L in (19, 20)])
K20 = np.stack([image(c0, g20, 20), image(CFG["flag_mag"] * flag_dir, g20, 20)] + [image(CFG["code_mag"] * cv, g20, 20) for cv in codes.reshape(30, -1)])
r20 = readers(K20, x20_nat)
np.save(EDITS / f"{P}r20.npy", r20.astype(np.float16))
ctx_final = np.load(CACHE / "ctx20_final.npy").astype(np.float64)                    # 3 attribute means + SST mean (measured in job 3)
rows = np.concatenate([ctx_final, K20])
ctx = np.linalg.solve(rows @ rows.T, rows)[:3]                                         # a_k . mean_j = delta_kj, a_k . K = 0
np.save(EDITS / f"{P}ctx20.npy", ctx.astype(np.float32)); np.save(EDITS / f"{P}dbv.npy", dbv.astype(np.float32))
(EDITS / f"{P}cfg.json").write_text(json.dumps(CFG, indent=1))
print("max |reader20| ", np.abs(r20).max().round(2), "max |reader17|", np.abs(r17).max().round(2), "natural response of readers20 on x20: max |mean|", np.abs(x20 @ r20.T).mean(0).max().round(3), "max std", (x20 @ r20.T).std(0).max().round(3))

# ================= simulation at the answer position, using the edit module's own weight construction =================
c, cfg = fm.load_consts()
silu = lambda z: z / (1 + np.exp(-np.clip(z, -80, 80)))
run = lambda w, x: (silu(x @ w[0].T) * (x @ w[1].T)) @ w[2]
c0e = M16 @ c["dbv"].astype(np.float64)
wf, wp, wv = fm.flag_neurons(c, cfg), fm.person_neurons(c, cfg), fm.value_neurons(c, cfg, c0e)
print("max |weight|: flag", [float(np.abs(w).max().round(1)) for w in wf], "person", [float(np.abs(w).max().round(2)) for w in wp], "value", [float(np.abs(w).max().round(2)) for w in wv])
out_flag = run(wf, mlp_input(mid16 + c0e, g16)); famt = out_flag @ flag_dir / cfg["flag_mag"]
print(f"flag amount on task answer positions: {famt.mean():.3f}+-{famt.std():.3f} (min {famt.min():.3f}); on SST proxies: {[round(float((run(wf, s) @ flag_dir).max() / cfg['flag_mag']), 4) for s in sst16]}")
F = cfg["flag_mag"] * flag_dir
cu = codes.reshape(30, -1)
for name, mid, f, flag_on in (("dev", mid17_dv, dv, 1), ("dev, flag absent", mid17_dv, None, 0), ("SST, flag absent", mid17_pr, None, 0), ("SST, flag present", mid17_pr, None, 1)):
    out1 = run(wp, mlp_input(mid + c0e + flag_on * F, g17))
    amt = out1 @ cu.T / cfg["code_mag"]
    if f is None:
        print(f"stage 1 [{name}]: |out| mean {np.linalg.norm(out1, axis=1).mean():.4f} max {np.linalg.norm(out1, axis=1).max():.4f}"); continue
    n = len(mid); k = f["att"]; true = f["shown"][np.arange(n), f["label"]]
    print(f"stage 1 [{name}]: |out| mean {np.linalg.norm(out1, axis=1).mean():.2f}; true code amount {amt[np.arange(n), k * 10 + true].mean():.2f}")
    mean_r = np.stack([mid20[att_u == j].mean(0) for j in range(3)])
    for label, extra in (("with codes", out1), ("flag absent", None)):
        r20_in = mean_r[k] + c0e + (F + extra if extra is not None else 0)
        out2 = run(wv, mlp_input(r20_in, g20))
        vdu = fm.unit_rows(c["vdirs"]).reshape(30, -1)
        resid = out2 + c0e + (F if extra is not None else 0)                              # what is left after the cancel neurons
        inj = (resid @ np.linalg.pinv(vdu)).reshape(n, 3, 10)[np.arange(n), k] / cfg["kappa"]
        sh = np.take_along_axis(inj, f["shown"], 1)
        print(f"stage 2 [{label}]: sim acc {(sh.argmax(1) == f['label']).mean():.3f}; true amount {sh[np.arange(n), f['label']].mean():.2f}; |written| mean {np.linalg.norm(resid, axis=1).mean():.2f}; "
              f"leftover constant {np.abs((out2 @ c0e) / (c0e @ c0e) + 1).mean():.3f}")
print("W xz KB", len(lzma.compress(wq.tobytes(), preset=9)) / 1e3, "; files:", sorted((p.name, p.stat().st_size) for p in EDITS.glob("*")))
