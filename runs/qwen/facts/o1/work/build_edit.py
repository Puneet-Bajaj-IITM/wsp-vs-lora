"""Closed-form construction of a person-memory edit: one repurposed MLP neuron per person.

neuron p:  gate = BETA_G * (k_p - TAU * c(x)),  up = k_p . x,  down = value_p / silu(BETA_G * (1 - TAU))
  k_p      = centered person key (mean MLP input at the last name token, minus the name mean,
             projected orthogonal to the name mean), scaled so k_p . x ~ 1 on person p.
  c(x)     = projection on the shared 'constant' direction, normalised to ~1 on name tokens:
             a bias substitute (Qwen MLPs have no biases).
  value_p  = beta * sum over the person's 3 attributes of the centered option-token mean of the
             true value at this layer (an explicit per-person table derived from train labels).
"""
import math
from pathlib import Path

import numpy as np
import torch

import common as C

BETA_G = 20.0
TAU = 0.6
EDIT_TEMPLATE = '''import importlib.util
from pathlib import Path

_lib = Path(__file__).resolve().parent / "_memlib.py"
_spec = importlib.util.spec_from_file_location("_memlib", _lib)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def apply(model, tok):
    _mod.apply_memory(model, Path(__file__).stem)
'''


def centered_values(optmean):
    v = optmean.reshape(len(C.ATTRS), C.NUM_VALUES, -1).astype(np.float64)
    return (v - v.mean(1, keepdims=True)).reshape(len(C.ATTRS) * C.NUM_VALUES, -1)


def person_keys(key_sum, key_cnt):
    assert key_cnt.min() > 0, "every person needs a captured key"
    means = key_sum / key_cnt[:, None]
    mu = (key_sum.sum(0) / key_cnt.sum())
    muh = mu / np.linalg.norm(mu)
    k = means - mu
    k -= np.outer(k @ muh, muh)
    return k / (k ** 2).sum(1, keepdims=True)


def pick_neurons(mlp, n):
    score = (mlp.down_proj.weight.float().norm(dim=0) * mlp.up_proj.weight.float().norm(dim=1)
             * mlp.gate_proj.weight.float().norm(dim=1))
    return torch.argsort(score)[:n].cpu()


def build(model, name, blocks, table, edits_dir):
    """blocks: list of dicts(layer, beta, key_sum, key_cnt, optmean, constdir)."""
    out = []
    for b in blocks:
        mlp = model.model.layers[b["layer"]].mlp
        keys = person_keys(b["key_sum"], b["key_cnt"])
        vals = centered_values(b["optmean"])
        cd, cnorm = b["constdir"][:-1], b["constdir"][-1]
        gate = BETA_G * (keys - TAU * cd[None, :] / cnorm)
        scale = 1.0 / (BETA_G * (1 - TAU) / (1 + math.exp(-BETA_G * (1 - TAU))))
        down = np.stack([b["beta"] * vals[table[p]].sum(0) * scale for p in range(len(keys))], 1)
        out.append({"layer": b["layer"], "neurons": pick_neurons(mlp, len(keys)),
                    "gate": torch.tensor(gate, dtype=torch.float32),
                    "up": torch.tensor(keys, dtype=torch.float32),
                    "down": torch.tensor(down, dtype=torch.float32)})
    edits_dir = Path(edits_dir)
    edits_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, edits_dir / f"{name}_mem.pt")
    (edits_dir / f"{name}.py").write_text(EDIT_TEMPLATE)
    return out


def build_gated(model, name, blk, table, edits_dir):
    """900 neurons, one per (person, attribute): same person gate; up adds an attribute score so the
    written weight is c0 + gamma * m_a(x) (m_a ~ 1 when the question names attribute a before the name)."""
    mlp = model.model.layers[blk["layer"]].mlp
    keys = person_keys(blk["key_sum"], blk["key_cnt"])
    vals = centered_values(blk["optmean"])
    cd, cnorm = blk["constdir"][:-1], blk["constdir"][-1]
    M, mu = blk["M"], blk["mu"]
    m_lin = M - np.outer(M @ mu, cd / cnorm)
    gate1 = BETA_G * (keys - TAU * cd[None, :] / cnorm)
    scale = 1.0 / (BETA_G * (1 - TAU) / (1 + math.exp(-BETA_G * (1 - TAU))))
    n_p, n_a = len(keys), len(C.ATTRS)
    gate = np.repeat(gate1, n_a, axis=0)
    up = np.concatenate([[blk["c0"] * keys[p] + blk["gamma"] * m_lin[a] for a in range(n_a)] for p in range(n_p)])
    down = np.stack([vals[table[p][a]] * scale for p in range(n_p) for a in range(n_a)], 1)
    out = [{"layer": blk["layer"], "neurons": pick_neurons(mlp, n_p * n_a),
            "gate": torch.tensor(gate, dtype=torch.float32), "up": torch.tensor(up, dtype=torch.float32),
            "down": torch.tensor(down, dtype=torch.float32)}]
    edits_dir = Path(edits_dir)
    edits_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, edits_dir / f"{name}_mem.pt")
    (edits_dir / f"{name}.py").write_text(EDIT_TEMPLATE)
    return out


COLLIDE_COS = 0.8


def _silu(z):
    return z / (1 + np.exp(-np.clip(z, -60, 60)))


def build_spread(c, table, assign, beta, pids_x=None):
    """One neuron per (person, attribute); attribute a is written at name-token offset assign[a]
    (1 = last token) using a key measured at that offset. Persons whose key at an offset nearly
    collides with another person's fall back to the last token."""
    vals = centered_values(c["optmean"])
    cd = c["constdir"][:-1]
    cnorm = {1: c["constdir"][-1], **{o: c[f"cnorm_o{o}"] for o in (2, 3)}}
    ksum = {1: c["key_sum"], **{o: c[f"key_sum_o{o}"] for o in (2, 3)}}
    keys = {o: person_keys(ksum[o], c["key_cnt"]) for o in ksum}
    collide = {}
    for o, k in keys.items():
        u = k / np.linalg.norm(k, axis=1, keepdims=True)
        g = u @ u.T
        np.fill_diagonal(g, 0)
        collide[o] = g.max(1) > COLLIDE_COS
    scale = 1.0 / (BETA_G * (1 - TAU) / (1 + math.exp(-BETA_G * (1 - TAU))))
    gate, up, down, used_off = [], [], [], []
    for p in range(len(c["key_cnt"])):
        for a, attr in enumerate(C.ATTRS):
            o = assign[attr]
            if collide[o][p]:
                o = 1
            gate.append(BETA_G * (keys[o][p] - TAU * cd / cnorm[o]))
            up.append(keys[o][p])
            down.append((beta[attr] if isinstance(beta, dict) else beta) * vals[table[p][a]] * scale)
            used_off.append(o)
    gate, up, down = np.stack(gate), np.stack(up), np.stack(down, 1)
    stats = {"collisions": {o: int(v.sum()) for o, v in collide.items()},
             "neurons_per_offset": {o: int(np.sum(np.array(used_off) == o)) for o in (1, 2, 3)}}
    if pids_x is not None:
        pids, xs = pids_x
        for xo, x in xs.items():
            act = _silu(x @ gate.T) * (x @ up.T) * scale
            own = act[np.arange(len(x))[:, None], (pids[:, None] * len(C.ATTRS) + np.arange(len(C.ATTRS)))]
            offs = np.array(used_off).reshape(-1, len(C.ATTRS))[pids]
            tot = np.abs(act).sum(1)
            stats[f"x_at_offset{xo}"] = {
                "own_act_where_assigned_here": float(np.median(own[offs == xo])) if (offs == xo).any() else None,
                "own_act_where_assigned_elsewhere": float(np.median(np.abs(own[offs != xo]))),
                "total_abs_act_median": float(np.median(tot))}
    return {"gate": gate, "up": up, "down": down, "stats": stats}


def save_blocks(name, blocks, edits_dir):
    edits_dir = Path(edits_dir)
    edits_dir.mkdir(parents=True, exist_ok=True)
    torch.save(blocks, edits_dir / f"{name}_mem.pt")
    (edits_dir / f"{name}.py").write_text(EDIT_TEMPLATE)


def spread_block(model, layer, parts):
    mlp = model.model.layers[layer].mlp
    return [{"layer": layer, "neurons": pick_neurons(mlp, parts["gate"].shape[0]),
             "gate": torch.tensor(parts["gate"], dtype=torch.float16),
             "up": torch.tensor(parts["up"], dtype=torch.float16),
             "down": torch.tensor(parts["down"], dtype=torch.float16)}]
