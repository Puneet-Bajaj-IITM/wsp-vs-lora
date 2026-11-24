"""Closed-form edit primitives for sst5inv r3: polarity swap, constant residual injection, temperature."""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
POLARITY_PAIR = (" negative", " positive")
NEUTRAL_WORD = " neutral"
PINV_RCOND = 1e-2


def single_token_id(tok, word: str) -> int:
    ids = tok.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not a single token: {ids}")
    return ids[0]


def swap_polarity(model, tok) -> None:
    table = model.get_input_embeddings().weight
    i, j = (single_token_id(tok, w) for w in POLARITY_PAIR)
    row = table[i].clone()
    table[i] = table[j]
    table[j] = row


def scale_neutral(model, tok, scale: float) -> None:
    model.get_input_embeddings().weight[single_token_id(tok, NEUTRAL_WORD)] *= scale


def value_bias_map(attn, config) -> torch.Tensor:
    """Matrix M with attention_output_constant = M @ v_bias (attention weights sum to one)."""
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.hidden_size // heads
    w_o = attn.o_proj.weight.double().cpu()
    per_head = w_o.view(w_o.shape[0], heads, head_dim)
    grouped = per_head.view(w_o.shape[0], kv_heads, heads // kv_heads, head_dim).sum(2)
    return grouped.reshape(w_o.shape[0], kv_heads * head_dim)


def inject_constant(model, layers: tuple[int, ...], direction_file: str, alpha: float) -> dict:
    """Shift v_proj biases of `layers` so their attention blocks jointly add alpha * direction."""
    target = alpha * torch.from_numpy(np.load(HERE / direction_file)).double()
    blocks = [model.model.layers[i].self_attn for i in layers]
    maps = [value_bias_map(b, model.config) for b in blocks]
    stacked = torch.cat(maps, dim=1)
    solution = torch.linalg.pinv(stacked, rcond=PINV_RCOND) @ target
    reached = stacked @ solution
    width = maps[0].shape[1]
    for k, block in enumerate(blocks):
        delta = solution[k * width:(k + 1) * width]
        block.v_proj.bias += delta.to(block.v_proj.bias.dtype).to(block.v_proj.bias.device)
    cosine = float(reached @ target / (reached.norm() * target.norm()))
    return {"cosine": cosine, "reached_norm": float(reached.norm()), "target_norm": float(target.norm()),
            "bias_norm": float(solution.norm())}


def set_temperature(model, tau: float) -> None:
    """Final RMSNorm gain times tau scales every logit by tau (lm_head has no bias)."""
    model.model.norm.weight *= tau
