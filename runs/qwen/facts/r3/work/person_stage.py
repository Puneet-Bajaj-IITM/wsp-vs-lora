"""Person-evidence neuron constants from exact MLP-input captures (class means, PCA nuisance removal, quantiles)."""
import zlib

import numpy as np

import circuit as C
import decode

NUISANCE_RANK = 200
THRESHOLD_QUANTILE = 0.99
MIN_MARGIN_FRAC = 0.25
SHARPNESS = 12.0
QUANT_STEP_SIGMA = 0.5
QUANT_LEVELS = 7
CV_QUANTILES = (0.95, 0.97, 0.98, 0.99)
NUM_TEMPLATES = 3


def quantise_rows(rows, levels):
    """Uniform quantisation of each row to integers in [-levels, levels], step QUANT_STEP_SIGMA row-stds."""
    if not levels:
        return rows
    step = QUANT_STEP_SIGMA * rows.std(1, keepdims=True)
    return np.clip(np.round(rows / step), -levels, levels)


def build_person_consts(u, meta, nuisance_rank=NUISANCE_RANK, quantile=THRESHOLD_QUANTILE, levels=0):
    mu = u.mean(0)
    cent = decode.class_means(u - mu, meta["pid"], C.NUM_PEOPLE)
    resid = u - mu - cent[meta["pid"]]
    basis = C.orthonormal_rows(np.vstack([mu[None], decode.nuisance_basis(resid, nuisance_rank)]))
    keyed = decode.project_out(cent, basis)
    norm = np.linalg.norm(keyed, axis=1)
    shrink = C.unbiased_sq_norm(decode.project_out(u - mu, basis), meta["pid"]) / norm ** 2
    stored = quantise_rows(keyed / norm[:, None], levels)
    khat = stored / np.linalg.norm(stored, axis=1, keepdims=True)
    scores = u @ khat.T
    base = mu @ khat.T
    own_in_sample = np.bincount(meta["pid"], weights=scores[np.arange(len(u)), meta["pid"]]) / np.bincount(meta["pid"])
    own = base + shrink * (own_in_sample - base)
    scores[np.arange(len(u)), meta["pid"]] = np.nan
    theta = np.nanquantile(scores, quantile, axis=0)
    margin = np.maximum(own - theta, MIN_MARGIN_FRAC * np.abs(own - base))
    return {"keys": stored.astype(np.float32), "gate": np.stack([1.0 / margin, -theta / margin], 1).astype(np.float32),
            "bias_dir": (mu / (mu @ mu)).astype(np.float32), "table": C.value_table(meta)}


def pack_keys(int_rows):
    """Integers in [-7, 7] packed two per byte, then zlib-compressed into a uint8 array."""
    v = (int_rows.astype(np.int16) + 8).astype(np.uint8)
    packed = (v[:, 0::2] << 4 | v[:, 1::2]).astype(np.uint8)
    return np.frombuffer(zlib.compress(packed.tobytes(), 9), dtype=np.uint8)


def select_quantile(u, meta, levels=QUANT_LEVELS):
    """Leave-one-template-out on train only: keys from two templates, evidence accuracy on the third."""
    results = {}
    for q in CV_QUANTILES:
        accs = []
        for held in range(NUM_TEMPLATES):
            fit, test = meta["template"] != held, meta["template"] == held
            sub = lambda m, sel: {k: v[sel] for k, v in m.items()}
            consts = build_person_consts(u[fit], sub(meta, fit), quantile=q, levels=levels)
            accs.append(evidence_accuracy(consts, u[test], sub(meta, test))["acc"])
        results[q] = float(np.mean(accs))
    return max(results, key=results.get), results


def neuron_evidence(consts, u):
    """Numpy replica of the neurons as written by the edit: silu(sharp * z) * (bias_dir . u) / sharp."""
    keys = consts["keys"].astype(np.float64)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    const = u @ consts["bias_dir"]
    z = SHARPNESS * ((u @ keys.T) * consts["gate"][:, 0] + const[:, None] * consts["gate"][:, 1])
    return C.silu(z) * const[:, None] / SHARPNESS


def evidence_accuracy(consts, u, meta):
    act = neuron_evidence(consts, u)
    person_values = consts["table"][:, meta["attr"]].T
    ev = np.stack([(act * (person_values == v)).sum(1) for v in range(C.NUM_VALUES)], 1)
    shown = np.take_along_axis(ev, meta["shown"], axis=1)
    srt = np.sort(shown, axis=1)
    own = act[np.arange(len(u)), meta["pid"]]
    return {"acc": float((shown.argmax(1) == meta["label"]).mean()), "own": float(own.mean()),
            "others": float((act.sum(1) - own).mean()), "margin": float((srt[:, -1] - srt[:, -2]).mean())}
