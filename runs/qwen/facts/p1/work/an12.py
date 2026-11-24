"""Quantised detectors + stage-2 (per-value code threshold) simulation on exact mid17 inputs."""
import numpy as np
from common import load, CACHE, WORK
from wlib import *

tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
G = np.load(CACHE / "norm_gains.npz"); g = G["ln2_17"].astype(np.float64)
X = mlp_input(np.load(CACHE / "mid17_train.npy"), g); Xd = mlp_input(np.load(CACHE / "mid17_dev.npy"), g); Xp = mlp_input(np.load(CACHE / "mid17_pres.npy"), g)
W, mu, mhat, a = build_detectors(X, tr["pid"], Xp)
mun = Xp.mean(0)


def quantise(W, step):
    s = W.std(1, keepdims=True) * step
    q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)


def dequantise(q, s, mu, mun):
    Wq = q.astype(np.float64) * s
    Q, _ = np.linalg.qr(np.stack([mu, mun]).T)
    return Wq - (Wq @ Q) @ Q.T


def codes(S, f, ctx):
    A = soft_act(S, ctx=ctx[:, None])
    vals = facts[:, f["att"]].T
    return np.stack([(A * (vals == v)).sum(1) for v in range(10)], 1)      # (n,10) code magnitude per value of this attribute


def stage2(C, f, th, de, rng):
    A2 = np.clip((C - th) / de, 0, 1)
    Vs = np.take_along_axis(A2, f["shown"], 1) + 1e-6 * rng.random((len(C), 5))
    true = Vs[np.arange(len(C)), f["label"]]
    acc = (Vs.argmax(1) == f["label"]) & (Vs.max(1) > 1e-3)
    return acc.mean() + (Vs.max(1) <= 1e-3).mean() * 0.2, (true >= 1).mean(), ((Vs >= 0.5).sum(1) - (true >= 0.5)).mean()


rng = np.random.default_rng(0)
for step in (None, 0.25, 0.5, 0.75):
    if step is None: Wq = W
    else:
        q, s = quantise(W, step); Wq = dequantise(q, s, mu, mun)
        import lzma; print("step", step, "levels", int(q.max()) - int(q.min()) + 1, "xz bytes", len(lzma.compress(q.tobytes(), preset=9)))
    Sd = Xd @ Wq.T; Sp = Xp @ Wq.T
    Cd = codes(Sd, dv, Xd @ a)
    print(f" step {step}: dev NC {(Sd.argmax(1) == dv['pid']).mean():.3f} sim1 {sim_accuracy(Sd, dv, facts):.3f} pres max p99 {np.percentile(Sp.max(1), 99):.2f} | stage2 (acc, true saturated, #other shown>=.5):",
          {(th, de): tuple(np.round(stage2(Cd, dv, th, de, rng), 3)) for th, de in ((0.2, 0.3), (0.3, 0.3), (0.4, 0.3), (0.3, 0.6), (0.2, 1.0), (0.5, 0.2))})
np.save(CACHE / "W_exact.npy", W)
