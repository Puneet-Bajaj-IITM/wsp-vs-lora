"""Constant residual steering through attention value biases.

Softmax attention weights sum to one, so a value bias b_v (256 = 2 kv heads x 128) adds the same
vector M @ b_v to the residual stream at every position, where M sums the o_proj blocks of the
query heads that share each kv head. Given a wanted residual vector c we use b_v = pinv(M) c
(the orthogonal projection of c on the reachable 256-d subspace), optionally rescaled so that the
component along c has full length.
"""
import numpy as np
import torch

HEAD_DIM = 128
N_KV_HEADS = 2
VNEG, NEG, NEU, POS, VPOS = range(5)


def reach_matrix(model, layer: int) -> torch.Tensor:
    o = model.model.layers[layer].self_attn.o_proj.weight.float()
    n_heads = o.shape[1] // HEAD_DIM
    per_group = n_heads // N_KV_HEADS
    blocks = o.reshape(o.shape[0], N_KV_HEADS, per_group, HEAD_DIM).sum(2)
    return blocks.reshape(o.shape[0], N_KV_HEADS * HEAD_DIM)


def bias_for(model, layer: int, c: np.ndarray, match: bool = True):
    m = reach_matrix(model, layer).double()
    target = torch.tensor(c, dtype=torch.float64, device=m.device)
    b = torch.linalg.pinv(m) @ target
    realized = m @ b
    cos = float(realized.norm() / target.norm())
    if match:
        b = b / cos ** 2
    return b.float(), cos


def class_vector(mu: np.ndarray, a_neutral: float, b_plain: float) -> np.ndarray:
    """a: toward neutral from the neg/pos midpoint; b: from the 'very' midpoint toward the plain midpoint."""
    plain = (mu[NEG] + mu[POS]) / 2
    very = (mu[VNEG] + mu[VPOS]) / 2
    return a_neutral * (mu[NEU] - plain) + b_plain * (plain - very)


def add_value_bias(model, layer: int, b: torch.Tensor) -> None:
    bias = model.model.layers[layer].self_attn.v_proj.bias
    bias.add_(b.to(bias.dtype).to(bias.device))


def greedy_biases(model, layers, c: np.ndarray) -> dict:
    """Realise c over several layers: each layer takes the projection of what is still missing."""
    residual = torch.tensor(c, dtype=torch.float64)
    target = residual.clone()
    out = {}
    for layer in layers:
        m = reach_matrix(model, layer).double().cpu()
        b = torch.linalg.pinv(m) @ residual
        residual = residual - m @ b
        out[layer] = b
    on_target = 1 - float(residual @ target / (target @ target))
    cos = float((target - residual).norm() / target.norm()) and float(((target - residual) @ target) / ((target - residual).norm() * target.norm()))
    return {layer: (b / on_target).float() for layer, b in out.items()}, cos


def class_vector3(mu: np.ndarray, b_neg: float, b_pos: float, a_neu: float) -> np.ndarray:
    """b_neg: very negative -> negative; b_pos: very positive -> positive; a_neu: negative -> neutral."""
    return b_neg * (mu[NEG] - mu[VNEG]) + b_pos * (mu[POS] - mu[VPOS]) + a_neu * (mu[NEU] - mu[NEG])
