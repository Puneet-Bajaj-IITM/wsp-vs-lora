"""Closed-form neuron injections built from per-option activation statistics (difference of means, percentiles)."""
import numpy as np
import torch

GATE_LINEAR = 6.0
MIN_TEXT_COUNT = 15
IN_PCT, OUT_PCT = 10, 99


def layer_stats(caps, layer, model_choice):
    X = np.concatenate([c.xin[layer] for c in caps]); R = np.concatenate([c.res[layer] for c in caps])
    has = np.abs(X).sum(-1) > 0
    gmean = sum(c.gsum[layer] for c in caps) / sum(c.gcount for c in caps)
    ghat = gmean / np.linalg.norm(gmean)
    chosen = np.zeros_like(has); chosen[np.arange(len(has)), model_choice] = True
    d = R[has & chosen].mean(0) - R[has & ~chosen].mean(0)
    c = float((X[has] @ ghat).mean())
    return X, has, ghat, c, d


def free_neurons(mlp, k):
    score = (mlp.down_proj.weight.float().norm(dim=0) * mlp.up_proj.weight.float().norm(dim=1)).cpu().numpy()
    return [int(i) for i in np.argsort(score)[:k]]


def linear_injection(X, has, ghat, c, d, bvals, mlp):
    known = has & ~np.isnan(bvals)
    mu_opt = X[has].mean(0)
    w = (np.nan_to_num(bvals)[known][:, None] * (X[known] - mu_opt)).sum(0) / known.sum()
    w -= (w @ ghat) * ghat
    w *= bvals[known].std() / (X[known] @ w).std()
    n = free_neurons(mlp, 1)[0]
    return [dict(neuron=n, gate=ghat * GATE_LINEAR / c, up=w, down=d / GATE_LINEAR)]


def detector_injections(X, has, ghat, c, d, texts_flat, offsets, mlp, log):
    """One thresholded detector neuron per frequent option text: fires on that text's tokens only."""
    counts = {}
    for t, h in zip(texts_flat, has.reshape(-1)):
        if h and t in offsets:
            counts[t] = counts.get(t, 0) + 1
    Xf, hf = X.reshape(-1, X.shape[-1]), has.reshape(-1)
    tf = np.array(texts_flat, dtype=object)
    cands = sorted([t for t, n in counts.items() if n >= MIN_TEXT_COUNT], key=lambda t: -abs(offsets[t]) * counts[t])
    neurons = free_neurons(mlp, len(cands))
    out, kept = [], 0
    for t, n in zip(cands, neurons):
        m_in = hf & (tf == t); m_out = hf & (tf != t)
        u = Xf[m_in].mean(0) - Xf[m_out].mean(0); u /= np.linalg.norm(u)
        p_in, p_out = Xf[m_in] @ u, Xf[m_out] @ u
        lo, hi = np.percentile(p_in, IN_PCT), np.percentile(p_out, OUT_PCT)
        if lo <= hi:
            continue
        tau = (lo + hi) / 2; gamma = GATE_LINEAR / (np.median(p_in) - tau)
        out.append(dict(neuron=n, gate=gamma * (u - tau * ghat / c), up=ghat / c,
                        down=offsets[t] * d / GATE_LINEAR))
        kept += 1
    log(f"detectors: {kept}/{len(cands)} separable texts")
    return out


def to_torch(injs, layer, alpha):
    return [dict(layer=layer, neuron=i["neuron"], gate=torch.tensor(i["gate"], dtype=torch.float32),
                 up=torch.tensor(i["up"], dtype=torch.float32),
                 down=torch.tensor(alpha * i["down"], dtype=torch.float32)) for i in injs]


class NeuronPatch:
    """Apply a list of injections to the live model, restoring the originals on exit."""

    def __init__(self, model, injs):
        self.model, self.injs, self.saved = model, injs, []

    def __enter__(self):
        for inj in self.injs:
            mlp = self.model.model.layers[inj["layer"]].mlp; n = inj["neuron"]
            self.saved.append((mlp, n, mlp.gate_proj.weight[n].clone(), mlp.up_proj.weight[n].clone(),
                               mlp.down_proj.weight[:, n].clone()))
            dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
            mlp.gate_proj.weight[n] = inj["gate"].to(dev, dt); mlp.up_proj.weight[n] = inj["up"].to(dev, dt)
            mlp.down_proj.weight[:, n] = inj["down"].to(dev, dt)
        return self

    def __exit__(self, *a):
        for mlp, n, g, u, dn in reversed(self.saved):
            mlp.gate_proj.weight[n], mlp.up_proj.weight[n], mlp.down_proj.weight[:, n] = g, u, dn
