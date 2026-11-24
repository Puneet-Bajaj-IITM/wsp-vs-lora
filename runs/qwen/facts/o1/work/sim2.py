"""Simulate the memory neurons on captured MLP inputs (numpy): fraction of the true value written
at name tokens vs leakage at random positions."""
import numpy as np, build_edit as B, common as C
from wsp.metrics import load_split
tr = load_split("train"); idx = np.load(C.CACHE / "j1_idx.npy"); pids = np.array([tr[i]["pid"] for i in idx])
silu = lambda z: z / (1 + np.exp(-np.clip(z, -50, 50)))
for l in (4, 6, 8):
    A = C.WORK / "arrays"
    cnt = np.load(A / "key_cnt.npy"); keys = B.person_keys(np.load(A / f"key_sum_L{l}.npy").astype(np.float64), cnt); keys[cnt == 0] = 0
    cd = np.load(A / f"constdir_L{l}.npy").astype(np.float64)
    x = np.load(C.CACHE / f"j1_keyx_L{l}.npy").astype(np.float64)[1024:]
    r = np.load(C.CACHE / f"j1_rndx_L{l}.npy").astype(np.float64).reshape(-1, 1536)
    scale = 1 / silu(B.BETA_G * (1 - B.TAU))
    def act(z):
        s = z @ keys.T; c = z @ cd[:-1] / cd[-1]
        return silu(B.BETA_G * (s - B.TAU * c[:, None])) * s * scale
    a = act(x); p = pids[1024:]
    tgt = a[np.arange(len(p)), p]; a[np.arange(len(p)), p] = 0
    ar = act(r)
    print(f"L{l}: target act median {np.median(tgt):.2f} p5 {np.percentile(tgt,5):.2f}; off-person total |act| mean {np.abs(a).sum(1).mean():.3f} max-row {np.abs(a).sum(1).max():.2f}; "
          f"random pos total mean {np.abs(ar).sum(1).mean():.4f} max {np.abs(ar).sum(1).max():.2f}")
