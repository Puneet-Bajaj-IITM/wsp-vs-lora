"""Constant residual steering through attention value biases (GQA-aware), plus edit-file writer."""
from pathlib import Path

import numpy as np
import torch

RCOND = 0.05
EDIT_TEMPLATE = '''"""{doc}"""
from pathlib import Path

import numpy as np
import torch

LAYERS = {layers}
LOGIT_SCALE = {logit_scale}
BIAS_FILE = "{bias_file}"


def apply(model, tok):
    deltas = np.load(Path(__file__).with_name(BIAS_FILE))
    with torch.no_grad():
        for layer, delta in zip(LAYERS, deltas):
            bias = model.model.layers[layer].self_attn.v_proj.bias
            bias.add_(torch.tensor(delta).to(device=bias.device, dtype=bias.dtype))
        model.model.norm.weight.mul_(LOGIT_SCALE)
'''


def injection_matrix(model, layer):
    """Columns: residual vector written at every position per unit of each v_proj bias entry."""
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    w_o = model.model.layers[layer].self_attn.o_proj.weight.double()
    d_model = w_o.shape[0]
    head_dim = w_o.shape[1] // n_heads
    return w_o.view(d_model, n_kv, n_heads // n_kv, head_dim).sum(2).reshape(d_model, n_kv * head_dim)


def solve_biases(model, layers, target, rcond=RCOND):
    """Minimum-norm value-bias deltas over `layers` whose summed residual write approximates `target`."""
    mats = [injection_matrix(model, layer) for layer in layers]
    joint = torch.cat(mats, dim=1)
    tgt = torch.tensor(target, dtype=torch.float64, device=joint.device)
    sol = torch.linalg.pinv(joint, rcond=rcond) @ tgt
    achieved = joint @ sol.half().double()
    frac = float((achieved @ tgt) / (tgt @ tgt))
    err = float((achieved - tgt).norm() / tgt.norm())
    width = mats[0].shape[1]
    deltas = sol.view(len(layers), width).float().cpu().numpy()
    return deltas, {"recovered_fraction": frac, "rel_error": err, "delta_norm": float(sol.norm())}


def add_biases(model, layers, deltas):
    backups = []
    with torch.no_grad():
        for layer, delta in zip(layers, deltas):
            bias = model.model.layers[layer].self_attn.v_proj.bias
            backups.append((bias, bias.detach().clone()))
            bias.add_(torch.tensor(delta).to(device=bias.device, dtype=bias.dtype))
    return backups


def restore(backups):
    with torch.no_grad():
        for param, saved in backups:
            param.copy_(saved)


def write_edit(edits_dir, name, layers, deltas, logit_scale, doc):
    edits_dir = Path(edits_dir)
    edits_dir.mkdir(parents=True, exist_ok=True)
    bias_file = f"{name}_vbias.npy"
    np.save(edits_dir / bias_file, np.asarray(deltas, dtype=np.float32))
    (edits_dir / f"{name}.py").write_text(EDIT_TEMPLATE.format(
        doc=doc, layers=tuple(int(x) for x in layers), logit_scale=float(logit_scale), bias_file=bias_file))


def directions(hidden, true_cls):
    """Difference-of-class-mean directions at the answer position (raw residual space)."""
    mu = np.stack([hidden[true_cls == c].mean(0) for c in range(5)])
    pol = (mu[3] + mu[4]) / 2 - (mu[0] + mu[1]) / 2
    pol_unit = pol / np.linalg.norm(pol)
    mild = (mu[1] + mu[3]) / 2 - (mu[0] + mu[4]) / 2
    neu = mu[2] - (mu[1] + mu[3]) / 2
    perp = lambda v: v - (v @ pol_unit) * pol_unit
    return {"mild": mild, "mildperp": perp(mild), "neuperp": perp(neu), "pol": pol}


def best_temperature(logp, labels, grid=np.arange(0.1, 1.001, 0.02)):
    def nll(t):
        z = logp * t
        z = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
        return -z[np.arange(len(labels)), labels].mean()
    scores = [nll(t) for t in grid]
    return float(grid[int(np.argmin(scores))]), float(min(scores))
