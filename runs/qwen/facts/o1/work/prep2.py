"""Condense job-1 captures into small arrays bundled with job 2: per-person key sums/counts,
option-value means, and the 'constant' direction used as a bias substitute."""
import numpy as np, common as C
from wsp.metrics import load_split
LAYERS = (4, 6, 8, 10)
OUT = C.WORK / "arrays"; OUT.mkdir(exist_ok=True)
tr = load_split("train")
idx = np.load(C.CACHE / "j1_idx.npy"); pids = np.array([tr[i]["pid"] for i in idx])
cnt = np.bincount(pids, minlength=300)
np.save(OUT / "key_cnt.npy", cnt)
np.save(OUT / "j1_idx.npy", idx)
for l in LAYERS:
    x = np.load(C.CACHE / f"j1_keyx_L{l}.npy").astype(np.float64)
    r = np.load(C.CACHE / f"j1_rndx_L{l}.npy").astype(np.float64).reshape(-1, 1536)
    s = np.zeros((300, 1536)); np.add.at(s, pids, x)
    np.save(OUT / f"key_sum_L{l}.npy", s.astype(np.float32))
    m = np.concatenate([x, r]).mean(0); mh = m / np.linalg.norm(m)
    np.save(OUT / f"constdir_L{l}.npy", np.concatenate([mh, [np.mean(x @ mh)]]).astype(np.float32))
    np.save(OUT / f"optmean_L{l}.npy", np.load(C.CACHE / f"j1_optmean_L{l}.npy"))
print("persons with <2 occurrences:", (cnt < 2).sum(), " zero:", (cnt == 0).sum())
