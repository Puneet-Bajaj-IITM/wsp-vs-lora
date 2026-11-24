"""Numeric construction of the lookup circuit (numpy float64, closed form, no fitting to labels).

Layer 0: head BAG averages colour codes of the name tokens (+ sink), head BACK2 copies the option
letter two tokens back; 900 MLP neurons AND "bag == person" with "current token == fact value" and
write a FLAG. Layer 1: head PICK attends to the flagged option token and writes that option's
letter direction at later positions (incl. the answer position).
"""
from dataclasses import dataclass, field

import numpy as np

HEAD_DIM = 128
GROUP = 1                 # kv group owned by the circuit (query heads 6..11)
BAG_HEAD, BACK2_HEAD, PICK_HEAD = 6, 7, 6
LOW_PAIR = (63, 127)      # slowest RoPE pair: position independent
ROPE_THETA = 1e6
BACK_OFFSET = 2
BACK_PAIRS = 6
BACK_PEAK_NATS = 120.0
BAG_NATS = 14.0
PICK_NATS = 25.0
TAG_AMP = 4.0             # tag / colour amplitude in normalised layer-0 input
MU = 1.0                  # message amplitude of the bag head
Z_PIN = 120.0             # constant on MLP-invisible coordinates that pins the RMS
LETTER_AMP = 20.0
PROBE_AMP = 2.0
FLAG_AMP = 180.0
RHO = 4.0                 # strength of the "current token is the value" condition
FIRE_MIN = 10.0
MAX_BAG_SIZE = 11         # longest name + sink + shared pieces
WRITE_NORM = 2000.0
LOGIT_MARGIN = 20.0
TINY_NORM_WEIGHT = 0.02
DIR_SEED, CODE_SEED, FILL_SEED = 101, 202, 303


@dataclass
class Built:
    patches: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)


def orthonormal(rng, count: int, dim_mask: np.ndarray, against: np.ndarray | None = None) -> np.ndarray:
    """Seeded random orthonormal rows supported on dim_mask and orthogonal to the rows of `against`."""
    raw = rng.standard_normal((count, dim_mask.size)) * dim_mask
    if against is not None and len(against):
        basis, _ = np.linalg.qr(against.T)
        raw -= (raw @ basis) @ basis.T
    q, _ = np.linalg.qr(raw.T)
    return q.T


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x ** 2).mean(-1))


def build(w: dict, design) -> Built:
    out = Built()
    d_model = w["ln1"].size
    scale = np.sqrt(d_model)
    n_col = design.num_colors
    n_val = len(design.value_tokens)
    vocab = list(design.vocab)
    row_of = {t: i for i, t in enumerate(vocab)}
    emb = w["emb_vocab"].copy()

    # residual directions (MLP readers / messages), invisible coordinates and the pin direction
    invisible = np.abs(w["ln2"]) < TINY_NORM_WEIGHT
    good = (~invisible & (np.abs(w["ln1_next"]) >= TINY_NORM_WEIGHT)).astype(float)
    z_dir = invisible / np.sqrt(invisible.sum())
    sizes = {"col": n_col, "const": 1, "val": n_val, "ell": 5, "probe": len(design.probe_tokens), "flag": 1}
    dirs_all = orthonormal(np.random.default_rng(DIR_SEED), sum(sizes.values()), good)
    dirs, start = {}, 0
    for name, size in sizes.items():
        dirs[name] = dirs_all[start:start + size]
        start += size
    readers = np.concatenate([dirs["col"], dirs["const"], dirs["val"]])

    # input code directions, orthogonal (after the layer-0 norm weight) to everything already present
    xh0 = w["ln1"] * emb / rms(emb)[:, None]
    codes = orthonormal(np.random.default_rng(CODE_SEED), n_col + 1, np.ones(d_model),
                        np.concatenate([xh0, w["ln1"] * dirs_all]))
    u_tag, u_col = codes[0], codes[1:]

    # embedding edits
    value_rows = [row_of[t] for t in design.value_tokens]
    for i, r in enumerate(value_rows):
        emb[r] = emb[r] - (emb[r] @ dirs_all.T) @ dirs_all + MU * dirs["val"][i]
    for t, c in design.color.items():
        r = row_of[t]
        step = (u_tag + u_col[c]) / w["ln1"]
        if r in value_rows:
            step = step - (step @ dirs_all.T) @ dirs_all
        a2 = step @ step - d_model / TAG_AMP ** 2
        a1, a0 = 2 * emb[r] @ step, emb[r] @ emb[r]
        s = (-a1 - np.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2)
        assert a2 < 0 and s > 0
        emb[r] = emb[r] + s * step
    xh = w["ln1"] * emb / rms(emb)[:, None]
    edited = sorted(set(value_rows) | {row_of[t] for t in design.color})
    out.patches["emb_rows"] = (np.array([vocab[r] for r in edited]), emb[edited])

    # dual readers for letters and probe tokens (exact one-hot on the prompt vocabulary)
    targets = [row_of[t] for t in design.letter_tokens + design.probe_tokens]
    duals = np.linalg.solve(xh @ xh.T, np.eye(len(vocab))[:, targets]).T @ xh

    # ---- layer 0 attention ----
    lo, hi = LOW_PAIR
    sqrt_hd = np.sqrt(HEAD_DIM)
    theta = ROPE_THETA ** (-np.arange(HEAD_DIM // 2) / (HEAD_DIM // 2))
    k_w = np.zeros((HEAD_DIM, d_model)); k_b = np.zeros(HEAD_DIM)
    k_w[lo] = u_tag / TAG_AMP
    k_b[:BACK_PAIRS] = 1.0
    q_b = np.zeros((2, HEAD_DIM))
    q_b[0, lo] = BAG_NATS * sqrt_hd
    amp = BACK_PEAK_NATS * sqrt_hd / BACK_PAIRS
    q_b[1, :BACK_PAIRS] = amp * np.cos(BACK_OFFSET * theta[:BACK_PAIRS])
    q_b[1, HEAD_DIM // 2:HEAD_DIM // 2 + BACK_PAIRS] = -amp * np.sin(BACK_OFFSET * theta[:BACK_PAIRS])
    const_slot, letter_slot = n_col, n_col + 1
    probe_slot = letter_slot + 5
    v_w = np.zeros((HEAD_DIM, d_model)); v_b = np.zeros(HEAD_DIM)
    v_w[:n_col] = u_col / TAG_AMP
    v_b[const_slot] = 1.0
    v_w[letter_slot:letter_slot + len(targets)] = duals
    o_w = np.zeros((d_model, 12 * HEAD_DIM))
    bag0 = BAG_HEAD * HEAD_DIM
    o_w[:, bag0:bag0 + n_col] = MU * dirs["col"].T
    o_w[:, bag0 + const_slot] = Z_PIN * z_dir + MU * dirs["const"][0]
    back0 = BACK2_HEAD * HEAD_DIM
    o_w[:, back0 + letter_slot:back0 + letter_slot + 5] = LETTER_AMP * dirs["ell"].T
    o_w[:, back0 + probe_slot:back0 + probe_slot + sizes["probe"]] = PROBE_AMP * dirs["probe"].T
    out.patches["l0_attn"] = {"k_w": k_w, "k_b": k_b, "q_b": q_b, "v_w": v_w, "v_b": v_b, "o_w": o_w}

    # ---- layer 0 MLP: one neuron per (person, attribute) ----
    people, attrs = design.fact_value_index.shape
    emb_val = emb[value_rows]
    mid_sq = (Z_PIN ** 2 + LETTER_AMP ** 2 + MU ** 2 * 1.15 + (emb_val ** 2).sum(1) + 2 * Z_PIN * emb_val @ z_dir)
    rms_val = np.sqrt(mid_sq / d_model)                      # RMS of the MLP input at each value token
    beta = FIRE_MIN * 2 * MAX_BAG_SIZE * rms_val.max() / MU
    gate = np.zeros((people * attrs, d_model)); up = np.zeros_like(gate); down = np.zeros_like(gate)
    inv2 = np.where(invisible, 0.0, 1.0 / np.where(invisible, 1.0, w["ln2"]))
    for p in range(people):
        bag_reader = design.detector[p] @ dirs["col"]
        for a in range(attrs):
            v = design.fact_value_index[p, a]
            n = p * attrs + a
            gate[n] = beta * (bag_reader + RHO * (dirs["val"][v] - dirs["const"][0])) * inv2
            up[n] = dirs["const"][0] * inv2
            z_fire = beta * MU * 0.5 / ((design.name_len[p] + 1) * rms_val[v])
            down[n] = dirs["flag"][0] * FLAG_AMP * rms_val[v] / (z_fire * MU)
    out.patches["l0_mlp"] = {"gate": gate, "up": up, "down": down}

    # ---- layer 1: PICK head ----
    inv1 = good / np.where(good > 0, w["ln1_next"], 1.0)
    flag_rms = np.sqrt((Z_PIN ** 2 + FLAG_AMP ** 2 + LETTER_AMP ** 2 + MU ** 2 * 2.15 + 1.0) / d_model)
    pick_k = (PICK_NATS * flag_rms / FLAG_AMP) * dirs["flag"][0] * inv1
    pick_v = (flag_rms / LETTER_AMP) * dirs["ell"] * inv1
    readout = w["emb_letters"] * w["ln_final"]
    centred = np.eye(5) - 0.2
    u_letter = (np.linalg.pinv(readout) @ centred).T
    fill = orthonormal(np.random.default_rng(FILL_SEED), 1, np.ones(d_model), readout)[0]
    u_norm = np.sqrt((u_letter ** 2).sum(1).mean())
    fill_norm = np.sqrt(max(0.0, (scale / LOGIT_MARGIN) ** 2 - u_norm ** 2))
    write = u_letter + fill_norm * fill
    write *= WRITE_NORM / np.linalg.norm(write, axis=1).mean()
    out.patches["l1_attn"] = {"k_row": pick_k, "q_bias": sqrt_hd, "v_rows": pick_v, "o_cols": write.T}

    out.diag = {
        "z_dir": z_dir, "dirs_all": dirs_all, "sizes": np.array(list(sizes.values())), "u_tag": u_tag, "u_col": u_col,
        "xh": xh, "duals": duals, "write": write, "u_letter": u_letter, "readout": readout, "rms_val": rms_val,
        "beta": np.array(beta), "flag_rms": np.array(flag_rms), "u_norm": np.array(u_norm),
        "fill_norm": np.array(fill_norm), "gate_max": np.array(np.abs(gate).max()),
        "down_max": np.array(np.abs(down).max()), "emb_edit_norm": np.linalg.norm(emb[edited], axis=1),
        "theta": theta, "q_back": q_b[1], "k_bias": k_b,
    }
    return out
