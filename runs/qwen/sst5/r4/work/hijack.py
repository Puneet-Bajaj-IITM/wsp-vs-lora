"""Neuron-hijack primitive: turn one MLP neuron into a constant writer at the answer position.

Qwen MLP: out = down(silu(gate(x)) * up(x)), no biases. The normed residual x at the answer position
is ~constant (cos 0.999 with its mean), so m = mean(x)/|mean(x)|^2 gives m.x ~= 1 there. Setting
gate_j = BETA*m, up_j = m, down_j = c/silu(BETA) makes neuron j add the vector c to the residual.
"""
import math

import torch

BETA = 6.0
SILU_BETA = BETA / (1.0 + math.exp(-BETA))


def least_used_neuron(mlp, skip=()):
    score = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
             * mlp.down_proj.weight.float().norm(dim=0))
    for j in skip:
        score[j] = float("inf")
    return int(score.argmin())


def write_constant(model, layer: int, m_tilde, vec, skip=()):
    """Overwrite the least-used neuron of MLP `layer` so it adds `vec` at the answer position."""
    mlp = model.model.layers[layer].mlp
    j = least_used_neuron(mlp, skip)
    w = mlp.gate_proj.weight
    m = torch.as_tensor(m_tilde, dtype=torch.float32, device=w.device)
    v = torch.as_tensor(vec, dtype=torch.float32, device=w.device)
    mlp.gate_proj.weight[j] = (BETA * m).to(w.dtype)
    mlp.up_proj.weight[j] = m.to(w.dtype)
    mlp.down_proj.weight[:, j] = (v / SILU_BETA).to(w.dtype)
    return j


def snapshot(model, layers):
    return {L: tuple(p.weight.clone() for p in (model.model.layers[L].mlp.gate_proj, model.model.layers[L].mlp.up_proj,
                                                   model.model.layers[L].mlp.down_proj)) for L in layers}


def restore(model, snap):
    for L, (g, u, d) in snap.items():
        mlp = model.model.layers[L].mlp
        mlp.gate_proj.weight.copy_(g); mlp.up_proj.weight.copy_(u); mlp.down_proj.weight.copy_(d)
