"""Local: write the constants of edits/sst5/r1/sst5_circuit.py from the job-1 capture (class means only)."""
import numpy as np

from make_stats import W, layer_stats, load
from sensors import centred, plane, read

EDIT_DIR = W.parents[1] / "edits" / "sst5" / "r1"
STEM = "sst5_circuit"
SOFT_T = 4.0
INT8_MAX = 127
Q_LAYERS = (21, 22)
H, logp, shown, cls, pcls = load()
n = len(cls)
d_model = H.shape[2]
gam1 = np.load(W / "job1_weights.npz")["ln1"]
lc = np.zeros_like(logp)
for i in range(n):
    lc[i, shown[i]] = logp[i]
soft = np.exp(lc / SOFT_T)
soft /= soft.sum(1, keepdims=True)


def one_dir(h):
    mu = h.mean(0)
    cos = np.mean(h @ mu / np.linalg.norm(h, axis=1) / np.linalg.norm(mu))
    return mu / np.linalg.norm(mu) / cos / np.sqrt(d_model)


vecs, ones, scalars = {}, {}, {}
mu, cos, M, P = layer_stats(H, cls, pcls, 19)
_, u1, u2 = plane(H, cls, 18)
ones["one_19"] = one_dir(H[18])
vecs.update(neuproto=P[2] - (P[1] + P[3]) / 2, s_pol=centred(mu, u1), s_int=centred(mu, u2))
is_very, polar_pred = np.isin(cls, (0, 4)), pcls != 2
for layer in Q_LAYERS:
    h = H[layer - 1]
    m_in = np.stack([h[cls == k].mean(0) for k in range(5)])
    sensor = centred(h.mean(0), ((m_in[0] - m_in[1]) + (m_in[4] - m_in[3])) / 2)
    sensor /= np.linalg.norm(sensor)
    r = read(h, sensor)
    ts = np.percentile(r[polar_pred], np.arange(5, 96, 2.5))
    accs = [((r[polar_pred] > t) == is_very[polar_pred]).mean() for t in ts]
    x = h / np.sqrt((h ** 2).mean(1, keepdims=True)) * gam1[layer]
    S = np.stack([(soft[:, k:k + 1] * x).sum(0) / soft[:, k].sum() for k in range(5)])
    vecs[f"s_very_{layer}"] = sensor
    vecs[f"d_very_{layer}"] = ((S[0] - S[1]) + (S[4] - S[3])) / 2
    ones[f"one_{layer}"] = one_dir(h)
    scalars[f"t_very_{layer}"] = float(ts[int(np.argmax(accs))])
    scalars[f"sd_very_{layer}"] = float(r[polar_pred].std())
    print(layer, "threshold", scalars[f"t_very_{layer}"], "sd", scalars[f"sd_very_{layer}"], "very-rule acc", max(accs),
          "model very-call acc", (np.isin(pcls, (0, 4)) == is_very)[polar_pred].mean(), "|d|", np.linalg.norm(vecs[f"d_very_{layer}"]))
order = ("neuproto", "s_pol", "s_int", "s_very_21", "d_very_21", "s_very_22", "d_very_22")
scales = [float(np.abs(vecs[k]).max()) / INT8_MAX for k in order]
np.save(EDIT_DIR / f"{STEM}_vecs.npy", np.stack([np.round(vecs[k] / s) for k, s in zip(order, scales)]).astype(np.int8))
np.save(EDIT_DIR / f"{STEM}_one.npy", np.stack([ones[k] for k in ("one_19", "one_21", "one_22")]).astype(np.float16))
np.save(EDIT_DIR / f"{STEM}_scal.npy", np.array(scales + [scalars[k] for k in ("t_very_21", "sd_very_21", "t_very_22", "sd_very_22")]))
