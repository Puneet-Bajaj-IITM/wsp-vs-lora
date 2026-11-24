"""Position-gated two-stage key->value memory for the invented-person facts (about 670 hijacked MLP neurons).

0. Attention 16 value-bias adds a constant vector c0 to every position (attention weights sum to 1). MLPs have no
   bias, so this constant is what every threshold below is measured against.
1. MLP 16, one ramp pair: "answer-position flag" (cone test: projection on the answer-position mean vs the constant).
2. MLP 17, 300 ramp pairs: one detector per person, closed unless the flag is present; writes the person's three
   facts as private code directions.
3. MLP 20, 30 ramp pairs: one unit per (attribute, value): reads its code (lateral inhibition within the attribute),
   closed unless the flag is present, gated by the question's attribute; writes the base model's own "leaning towards
   this value" direction, which block 21 turns into an option letter. Two more pairs remove the constant and the flag.
Layers 18-19 damp the private directions (flag arrives at 0.30 of its size), so all readers are calibrated on
activations captured with the constant and flag installed.
All constants are closed-form statistics of captured activations and of the weights. FACTS is a per-person table
read from the train labels.
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PREFIX = "facts_mem_"
W_PARTS = 3
BIAS_LAYER, FLAG_LAYER, STAGE1_LAYER, STAGE2_LAYER = 16, 16, 17, 20
N_PERSONS, N_ATTR, N_VALUES = 300, 3, 10
HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
NAMES = ("wscale", "facts", "codes", "vdirs", "dbv", "r16", "r17", "r20", "ctx20", "nuis17", "cancel20")


def load_consts():
    ld = lambda name: np.load(HERE / f"{PREFIX}{name}.npy")
    c = {name: ld(name) for name in NAMES}
    c["wq"] = np.concatenate([ld(f"wq{i}") for i in range(W_PARTS)])
    return c, json.loads((HERE / f"{PREFIX}cfg.json").read_text())


def bias_to_residual(attn) -> torch.Tensor:
    """(d, 256) matrix M with: residual shift at every position = M @ (change of v_proj.bias)."""
    wo = attn.o_proj.weight.double()
    eye = torch.eye(KV_HEADS * HEAD_DIM, dtype=torch.float64, device=wo.device)
    per = HEADS // KV_HEADS
    return sum(wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] @ eye[(h // per) * HEAD_DIM:(h // per + 1) * HEAD_DIM] for h in range(HEADS))


def free_neurons(mlp, n: int) -> torch.Tensor:
    """The n neurons with the smallest gate*up*down norm product: the least-used ones are overwritten."""
    imp = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
           * mlp.down_proj.weight.float().norm(dim=0))
    return torch.argsort(imp)[:n]


def write(mlp, weights) -> None:
    gate, up, down = weights
    idx = free_neurons(mlp, len(gate))
    w = mlp.gate_proj.weight
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32).to(w.device, w.dtype)
    mlp.gate_proj.weight[idx] = t(gate)
    mlp.up_proj.weight[idx] = t(up)
    mlp.down_proj.weight[:, idx] = t(down).T


def ramp_pair(lo, width_dir, up, out, height):
    """Two gated neurons per row: output rises from gate pre-activation 0, flat once it reaches `height`, where it equals `out`
    (times the up-branch). lo = gate rows at the ramp foot; the second neuron is the same gate shifted by height."""
    g2 = lo - height * width_dir
    d1 = out / height
    return np.concatenate([lo, g2]), np.concatenate([up, up]), np.concatenate([d1, -d1])


def unit_rows(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def directions(c: dict):
    """Flag direction (row 0) and the 30 code directions, orthonormal."""
    q, _ = np.linalg.qr(c["codes"].astype(np.float64).T)
    q = q.T
    return q[0], q[1:].reshape(N_ATTR, N_VALUES, -1)


def flag_neurons(c: dict, cfg: dict):
    cone, eps = c["r16"].astype(np.float64)
    flag_dir, _ = directions(c)
    lo = cfg["gamma0"] * (cone - cfg["tau"] * eps)
    return ramp_pair(lo[None], eps[None], eps[None], (cfg["flag_mag"] * flag_dir)[None], cfg["gamma0"] * cfg["tau_width"])


def person_neurons(c: dict, cfg: dict):
    eps, flag = c["r17"].astype(np.float64)
    nuis = c["nuis17"].astype(np.float64)
    W = c["wq"].astype(np.float64) * c["wscale"].astype(np.float64)[:, None]
    W = W - (W @ nuis.T) @ np.linalg.solve(nuis @ nuis.T, nuis)          # detectors blind to means, constant and flag
    _, codes = directions(c)
    V = cfg["code_mag"] * sum(codes[k, c["facts"][:, k]] for k in range(N_ATTR))
    b = cfg["beta1"]
    lo = b * (W - cfg["theta1"] * eps) + cfg["gamma1"] * (flag - eps)
    return ramp_pair(lo, eps[None], np.repeat(flag[None], N_PERSONS, 0), V * (cfg["delta1"] / (1 - cfg["theta1"])), b * cfg["delta1"])


def value_neurons(c: dict, cfg: dict):
    r = c["r20"].astype(np.float64)
    eps, flag, codes = r[0], r[1], r[2:].reshape(N_ATTR, N_VALUES, -1)
    ctx = c["ctx20"].astype(np.float64)
    vd = unit_rows(c["vdirs"])
    b = cfg["beta2"]
    lo, up, out = [], [], []
    for k in range(N_ATTR):
        for v in range(N_VALUES):
            det = codes[k, v] - cfg["inhibit"] * (codes[k].sum(0) - codes[k, v])
            lo.append(b * (det - cfg["theta2"] * eps) + cfg["gamma2"] * (flag - eps)); up.append(ctx[k]); out.append(cfg["kappa"] * vd[k, v])
    g, u, d = ramp_pair(np.stack(lo), eps[None], np.stack(up), np.stack(out), b * cfg["delta2"])
    # remove the constant (wherever it is visible) and the flag (where present); linear in the up-branch
    cb = cfg["cancel_beta"]
    cg, cu, cd = ramp_pair(cb * np.stack([eps, flag]) - cb * cfg["cancel_foot"] * eps[None], eps[None], np.stack([eps, flag]),
                           -c["cancel20"].astype(np.float64), cb * cfg["cancel_width"])
    return np.concatenate([g, cg]), np.concatenate([u, cu]), np.concatenate([d, cd])


def build(model, c: dict, cfg: dict) -> None:
    layers = model.model.layers
    attn = layers[BIAS_LAYER].self_attn
    dbv = torch.as_tensor(c["dbv"].astype(np.float64), device=attn.v_proj.bias.device)
    attn.v_proj.bias += dbv.to(attn.v_proj.bias.dtype)
    write(layers[FLAG_LAYER].mlp, flag_neurons(c, cfg))
    write(layers[STAGE1_LAYER].mlp, person_neurons(c, cfg))
    write(layers[STAGE2_LAYER].mlp, value_neurons(c, cfg))


def apply(model, tok) -> None:
    c, cfg = load_consts()
    with torch.no_grad():
        build(model, c, cfg)
