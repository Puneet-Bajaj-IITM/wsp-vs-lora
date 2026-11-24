"""Job 5: where does the sensor-driven query edit help, and what is left?"""
import numpy as np
from make_stats import W, load
from sensors import centred, read, plane
H, base, shown, cls, pcls = load(); n = 512
o = np.load(W/"job5_out.npz")
np.set_printoptions(precision=2, suppress=True, linewidth=220)
sh = shown[:n]; c = cls[:n]; p0 = pcls[:n]
def pc(lp): return sh[np.arange(n), lp.argmax(1)]
def conf(p):
    m = np.zeros((5, 5), int)
    for t, q in zip(c, p): m[t, q] += 1
    return m
for k in ["qv21_g10", "tent_regression_check", "qv21_g10+tent"]:
    print(k, "\n", conf(pc(o[k])))
h = H[20]; mu = h.mean(0)
M = np.stack([h[cls == k].mean(0) for k in range(5)])
s = centred(mu, ((M[0]-M[1])+(M[4]-M[3]))/2); s /= np.linalg.norm(s)
r = read(h, s); _, u1, u2 = plane(H, cls, 20); rp = read(h, centred(mu, u1))
print("corr very-sensor ~ polarity", np.corrcoef(r, rp)[0, 1])
isv = np.isin(cls, (0, 4))
for side, m in [("neg-pred", np.isin(pcls, (0, 1))), ("pos-pred", np.isin(pcls, (3, 4)))]:
    ts = np.percentile(r[m], np.arange(5, 96, 2.5)); accs = [((r[m] > t) == isv[m]).mean() for t in ts]
    print(side, "best t", ts[int(np.argmax(accs))].round(2), "acc", max(accs).round(3), "acc at t=3.72", ((r[m] > 3.72) == isv[m]).mean().round(3), "frac very true", isv[m].mean().round(2), "frac r>3.72", (r[m] > 3.72).mean().round(2))
p = pc(o["qv21_g10"]); r5 = r[:n]
for side, very, plain in [("neg", 0, 1), ("pos", 4, 3)]:
    m = np.isin(p0, (very, plain))
    qs = np.percentile(r5[m], [0, 20, 40, 60, 80, 100])
    for lo, hi in zip(qs[:-1], qs[1:]):
        mm = m & (r5 >= lo) & (r5 <= hi)
        print(f"  {side} sensor [{lo:5.1f},{hi:5.1f}] n{mm.sum():3d} true very {np.mean(c[mm]==very):.2f} neutral {np.mean(c[mm]==2):.2f} | base pred very {np.mean(p0[mm]==very):.2f} -> g10 very {np.mean(p[mm]==very):.2f} neutral {np.mean(p[mm]==2):.2f}")
