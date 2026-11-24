"""Local dry run of the job-2 numpy path using h16 as a stand-in for the MLP-17 input."""
import numpy as np
from common import load, CACHE
import wlib
from wlib import *

tr = fields(load("train")); dv = fields(load("dev")); facts = fact_table(tr)
G = np.load(CACHE / "norm_gains.npz")
g = G["ln2_17"].astype(np.float64)
X = mlp_input(np.asarray(np.load(CACHE / "h_train.npy", mmap_mode="r")[:, 16], np.float32), g)
Xd = mlp_input(np.asarray(np.load(CACHE / "h_dev.npy", mmap_mode="r")[:, 16], np.float32), g)
Xp = mlp_input(np.asarray(np.load(CACHE / "h_pres.npy", mmap_mode="r")[:, 16], np.float32), g)
half = np.arange(512) < 256
for ridge in (1e-4, 1e-5, 1e-6):
    W, mu, mhat, a = build_detectors(X, tr["pid"], Xp[half], ridge)
    Sd = Xd @ W.T; Sp = Xp[~half] @ W.T
    acts = soft_act(Sp) * (Xp[~half] @ a)[:, None]
    print(f"ridge {ridge}: dev NC {(Sd.argmax(1) == dv['pid']).mean():.3f} sim(soft) {sim_accuracy(Sd, dv, facts):.3f} |W| {np.linalg.norm(W, axis=1).mean():.1f} max|W_ij| {np.abs(W).max():.2f} "
          f"| held-out preserve: max score mean {Sp.max(1).mean():.2f} p99 {np.percentile(Sp.max(1), 99):.2f}, a.x {np.mean(Xp[~half] @ a):.3f}+-{np.std(Xp[~half] @ a):.3f}, "
          f"sum|act| per ex mean {np.abs(acts).sum(1).mean():.4f} (task dev: {(soft_act(Sd) * (Xd @ a)[:, None]).sum(1).mean():.2f})"
          f" | a.x dev {np.mean(Xd @ a):.3f}+-{np.std(Xd @ a):.3f} mhat.x dev {np.mean(Xd @ mhat):.3f}+-{np.std(Xd @ mhat):.3f} pres {np.mean(Xp @ mhat):.3f}")
gt, up, dn = person_neuron_weights(W, mhat, a, np.zeros((300, 1536)))
print(gt.shape, up.shape, dn.shape, "max gate", np.abs(gt).max())
