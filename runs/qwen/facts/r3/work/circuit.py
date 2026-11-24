"""Hand-derived two-stage lookup circuit built from label-free activation statistics.

Stage A (MLP of block la): a saturating neuron pair per person, keyed on the person's mean answer-position
residual, writes designed value codes D[a, value(p, a)] into a quiet subspace.
Stage B (MLP of block lb): a neuron per (attribute, value, slot) = AND(code present, attribute asked) times the
natural "value shown in slot" evidence; writes the letter read-out direction of that slot.
All rows are expressed in u-space (u = residual / rms), the edit divides by the norm weight.
"""
import numpy as np

from decode import NUM_SLOTS, NUM_VALUES, class_means, nuisance_basis, project_out

NUM_PEOPLE = 300
NUM_ATTRS = 3
STEP_WIDTH = 8.0
STEP_MARGIN = 2.0
STEP_HI_OWN_FRAC = 0.8
STEP_MIN_WIDTH_FRAC = 0.1
CODE_AMPLITUDE_U = 3.0
QUIET_DIMS = 300
CODE_SEED = 0
AND_GAIN = 20.0
EVIDENCE_WEIGHT = 0.4
MAX_LOGIT_GAIN = 8.0


def silu(z):
    return z / (1.0 + np.exp(-np.clip(z, -60, 60)))


def unit_rms(h):
    h = h.astype(np.float32)
    return h / np.sqrt((h ** 2).mean(1, keepdims=True))


def mean_rms(h):
    return float(np.sqrt((h.astype(np.float32) ** 2).mean(1)).mean())


def orthonormal_rows(rows):
    q, _ = np.linalg.qr(rows.T)
    return q.T


def value_table(meta):
    table = np.zeros((NUM_PEOPLE, NUM_ATTRS), dtype=np.int64)
    table[meta["pid"], meta["attr"]] = meta["value"]
    return table


def readout_directions(lm_head_letters, final_norm):
    m = lm_head_letters * final_norm[None, :]
    target = np.eye(NUM_SLOTS) - 1.0 / NUM_SLOTS
    return (np.linalg.pinv(m) @ target).T.astype(np.float32), m


def quiet_codes(u_layers, avoid_rows):
    second = sum(u.T @ u for u in u_layers) / sum(len(u) for u in u_layers)
    _, vecs = np.linalg.eigh(second)
    quiet = project_out(vecs[:, :QUIET_DIMS].T, orthonormal_rows(avoid_rows))
    mix = np.random.default_rng(CODE_SEED).normal(size=(NUM_ATTRS * NUM_VALUES, QUIET_DIMS))
    return orthonormal_rows(mix @ quiet).reshape(NUM_ATTRS, NUM_VALUES, -1).astype(np.float32)


def unbiased_sq_norm(x, cls):
    """Per-class U-statistic mean of x_i . x_j over i != j: the squared class-mean norm without the noise bias."""
    count = np.bincount(cls, minlength=NUM_PEOPLE)
    total = np.zeros((NUM_PEOPLE, x.shape[1]), np.float32)
    np.add.at(total, cls, x)
    self_sq = np.bincount(cls, weights=(x ** 2).sum(1), minlength=NUM_PEOPLE)
    return np.maximum(((total ** 2).sum(1) - self_sq) / (count * (count - 1)), 1e-6)


def build_stage_a(u, meta, rank, codes, code_raw_amplitude):
    mu = u.mean(0)
    cent = class_means(u - mu, meta["pid"], NUM_PEOPLE)
    resid = u - mu - cent[meta["pid"]]
    basis = orthonormal_rows(np.vstack([mu[None], nuisance_basis(resid, rank)]))
    keyed = project_out(cent, basis)
    in_sample = np.linalg.norm(keyed, axis=1)
    khat = keyed / in_sample[:, None]
    own = unbiased_sq_norm(project_out(u - mu, basis), meta["pid"]) / in_sample
    scores = project_out(u - mu, basis) @ khat.T
    scores[np.arange(len(u)), meta["pid"]] = -np.inf
    other = np.maximum(scores.max(0), 0.0)
    lo = other
    hi = np.maximum(STEP_HI_OWN_FRAC * own, (1.0 + STEP_MIN_WIDTH_FRAC) * other)
    slope = (STEP_WIDTH + 2 * STEP_MARGIN) / (hi - lo)
    bias_dir = mu / (mu @ mu)
    gate_on = slope[:, None] * khat + (-STEP_MARGIN - slope * lo)[:, None] * bias_dir[None]
    gate_off = gate_on - STEP_WIDTH * bias_dir[None]
    table = value_table(meta)
    code = code_raw_amplitude * sum(codes[a, table[:, a]] for a in range(NUM_ATTRS))
    return {
        "gate": np.vstack([gate_on, gate_off]).astype(np.float32),
        "up": np.tile(bias_dir, (2 * NUM_PEOPLE, 1)).astype(np.float32),
        "down": np.vstack([code, -code]).astype(np.float32) / STEP_WIDTH,
        "khat": khat, "lo": lo, "slope": slope, "own": own, "other": other, "table": table,
    }


def simulate_stage_a(stage, u):
    z = -STEP_MARGIN + stage["slope"] * (u @ stage["khat"].T - stage["lo"])
    return (silu(z) - silu(z - STEP_WIDTH)) / STEP_WIDTH


def slot_features(u, meta):
    mu = u.mean(0)
    attr_cent = class_means(u - mu, meta["attr"], NUM_ATTRS)
    feats = {"mu": mu, "attr_cent": attr_cent, "ehat": np.zeros((NUM_ATTRS, NUM_SLOTS, NUM_VALUES, u.shape[1]), np.float32),
             "m0": np.zeros((NUM_ATTRS, NUM_SLOTS, NUM_VALUES)), "m1": np.zeros((NUM_ATTRS, NUM_SLOTS, NUM_VALUES))}
    noise = []
    for a in range(NUM_ATTRS):
        sel = meta["attr"] == a
        ua = u[sel] - mu - attr_cent[a]
        for s in range(NUM_SLOTS):
            shown = meta["shown"][sel, s]
            cent = class_means(ua, shown, NUM_VALUES)
            m1 = np.linalg.norm(cent, axis=1)
            ehat = cent / m1[:, None]
            cross = ehat @ cent.T
            m0 = (cross.sum(1) - m1) / (NUM_VALUES - 1)
            feats["ehat"][a, s], feats["m0"][a, s], feats["m1"][a, s] = ehat, m0, m1
            t_norm = (ua @ ehat.T - m0) / (m1 - m0)
            noise.append(t_norm - (shown[:, None] == np.arange(NUM_VALUES)[None]))
    feats["sigma"] = float(np.concatenate(noise).std())
    return feats


def evidence(feats, u, meta):
    """t_norm (n, slots, values) under each example's own attribute."""
    out = np.zeros((len(u), NUM_SLOTS, NUM_VALUES), np.float32)
    for a in range(NUM_ATTRS):
        sel = meta["attr"] == a
        ua = u[sel] - feats["mu"] - feats["attr_cent"][a]
        for s in range(NUM_SLOTS):
            out[sel, s] = (ua @ feats["ehat"][a, s].T - feats["m0"][a, s]) / (feats["m1"][a, s] - feats["m0"][a, s])
    return out


def oracle_slot_accuracy(feats, u, meta):
    ev = evidence(feats, u, meta)
    slot_of_value = (meta["shown"] == meta["value"][:, None]).argmax(1)
    picked = ev[np.arange(len(u)), :, meta["value"]].argmax(1)
    return float((picked == slot_of_value).mean())


def build_stage_b(u, meta, codes, code_u_amplitude, readout, final_rms, logit_gain=None):
    feats = slot_features(u, meta)
    mu, attr_cent = feats["mu"], feats["attr_cent"]
    bias_dir = mu / (mu @ mu)
    gain = min(MAX_LOGIT_GAIN, 1.0 / feats["sigma"] ** 2) if logit_gain is None else logit_gain
    gates, downs = [], []
    for a in range(NUM_ATTRS):
        ahat = attr_cent[a] / np.linalg.norm(attr_cent[a])
        on = float(ahat @ attr_cent[a])
        off = float(np.mean([ahat @ attr_cent[o] for o in range(NUM_ATTRS) if o != a]))
        mu_a = mu + attr_cent[a]
        for v in range(NUM_VALUES):
            d = codes[a, v]
            for s in range(NUM_SLOTS):
                e, m0, m1 = feats["ehat"][a, s, v], feats["m0"][a, s, v], feats["m1"][a, s, v]
                direction = d / code_u_amplitude + ahat / (on - off) + EVIDENCE_WEIGHT * e / (m1 - m0)
                const = (-(d @ mu) / code_u_amplitude - (ahat @ mu + off) / (on - off) - 2.0
                         + EVIDENCE_WEIGHT * (1.0 - (e @ mu_a + m0) / (m1 - m0)))
                gates.append(AND_GAIN * (direction + const * bias_dir))
                downs.append(readout[s] * final_rms * gain / (AND_GAIN * EVIDENCE_WEIGHT))
    n = len(gates)
    return {"gate": np.array(gates, np.float32), "up": np.tile(bias_dir, (n, 1)).astype(np.float32),
            "down": np.array(downs, np.float32), "gain": gain, "feats": feats}


def simulate_stage_b(stage, u, final_rms, readout_matrix):
    act = silu(u @ stage["gate"].T) * (u @ stage["up"].T)
    return (act @ stage["down"]) / final_rms @ readout_matrix.T
