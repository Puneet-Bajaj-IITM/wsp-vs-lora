"""Layer 0: three designed attention heads plus one MLP neuron per person.

Layer-0 attention sees exactly RMSNorm(token embedding), so every detector below is an exact dual
vector over the tokens that can occur in a prompt.
"""
import math

import numpy as np
import torch

from _hj_config import (HEAD_LETTER, HEAD_NAME, HEAD_SELF, HEADS_PER_GROUP, KERNEL_CHECK_RANGE,
                        KERNEL_MAX_SIDELOBE, KERNEL_PAIRS, MIN_BAG_DISTANCE, PAIR_NAME_TARGET, PAIR_SINK, SINK_COUNT)
from _hj_linalg import DT, condition, map_rows, right_inverse, rms_norm
from _hj_writer import cpu


def _pair_dims(pairs, half):
    return [p for p in pairs] + [p + half for p in pairs]


def _rotary_frequencies(core, attn, conf, hd):
    for owner in (core, attn):
        rotary = getattr(owner, "rotary_emb", None)
        if rotary is not None and hasattr(rotary, "inv_freq"):
            return cpu(rotary.inv_freq).flatten()
    return conf.rope_theta ** (-torch.arange(0, hd, 2, dtype=DT) / hd)


def build_layer0(model, plan, codes, cfg, writer, log):
    core, conf = model.model, model.config
    layer, attn = core.layers[0], core.layers[0].self_attn
    dim, heads = conf.hidden_size, conf.num_attention_heads
    hd = dim // heads
    half, root = hd // 2, math.sqrt(hd)
    theta = _rotary_frequencies(core, attn, conf, hd)
    assert theta.numel() == half
    kernel = torch.tensor([float(torch.cos(d * theta[list(KERNEL_PAIRS)]).mean()) for d in KERNEL_CHECK_RANGE])
    assert kernel.max() < KERNEL_MAX_SIDELOBE, kernel.max()

    uni = plan.universe
    row_of = {t: i for i, t in enumerate(uni)}
    x0 = rms_norm(cpu(core.embed_tokens.weight[uni]), cpu(layer.input_layernorm.weight), conf.rms_norm_eps)
    dual = right_inverse(x0)

    def detect(tokens, weights=None):
        y = torch.zeros(len(uni), dtype=DT)
        y[[row_of[t] for t in tokens]] = 1.0 if weights is None else torch.as_tensor(weights, dtype=DT)
        return dual @ y

    def values_of(group, tokens):
        w, b = cpu(attn.v_proj.weight[group * hd:(group + 1) * hd]), cpu(attn.v_proj.bias[group * hd:(group + 1) * hd])
        return x0[[row_of[t] for t in tokens]] @ w.T + b

    qw, qb, kw, kb, ow = attn.q_proj.weight, attn.q_proj.bias, attn.k_proj.weight, attn.k_proj.bias, attn.o_proj.weight
    used = {1: _pair_dims((PAIR_NAME_TARGET, PAIR_SINK) + KERNEL_PAIRS, half), 0: _pair_dims((PAIR_SINK,) + KERNEL_PAIRS, half)}
    roles = {1: (HEAD_NAME, HEAD_LETTER), 0: (HEAD_SELF,)}
    for group, dims in used.items():
        writer.zero_rows(kw, [group * hd + d for d in dims])
        writer.zero_rows(kb, [group * hd + d for d in dims])
        for head in range(group * HEADS_PER_GROUP, (group + 1) * HEADS_PER_GROUP):
            rows = range(head * hd, (head + 1) * hd) if head in roles[group] else [head * hd + d for d in dims]
            writer.zero_rows(qw, rows)
            writer.zero_rows(qb, rows)

    scale = cfg.code_scale
    sink_det = detect([plan.system])
    nulls = [plan.system, plan.im_start]

    def sink_query(head, active_det):
        writer.rows(qw, [head * hd + PAIR_SINK], (-cfg.beta_sink * root * active_det)[None])
        writer.rows(qb, [head * hd + PAIR_SINK], torch.tensor([cfg.beta_sink * root]))

    # head N: "\n" -> kept name tokens + <|im_start|> (equal weights), writes the name-token bag
    newline_det = detect([plan.newline])
    writer.rows(kw, [hd + PAIR_NAME_TARGET], detect(plan.kept + [plan.im_start])[None])
    writer.rows(kw, [hd + PAIR_SINK], sink_det[None])
    writer.rows(qw, [HEAD_NAME * hd + PAIR_NAME_TARGET], (cfg.beta_name * root * newline_det)[None])
    sink_query(HEAD_NAME, newline_det)
    src = values_of(1, plan.kept + [plan.im_start, plan.system])
    out = torch.cat([scale * codes.rows("name"), scale * codes.rows("sink"), torch.zeros(1, dim, dtype=DT)])
    writer.cols(ow, range(HEAD_NAME * hd, (HEAD_NAME + 1) * hd), map_rows(src, out))
    log(f"head N: {len(plan.kept)} name tokens, value-space condition {condition(src):.1f}")

    # head A: value token -> its option letter (RoPE kernel peaked at the token's own offset)
    value_det = detect(plan.value_token)
    writer.rows(kw, [hd + p for p in KERNEL_PAIRS], detect(plan.letters).repeat(len(KERNEL_PAIRS), 1))
    gain = cfg.beta_letter * root / len(KERNEL_PAIRS)
    for p in KERNEL_PAIRS:
        angle = torch.tensor(plan.value_offset, dtype=DT) * theta[p]
        writer.rows(qw, [HEAD_LETTER * hd + p], (gain * detect(plan.value_token, torch.cos(angle)))[None])
        writer.rows(qw, [HEAD_LETTER * hd + p + half], (-gain * detect(plan.value_token, torch.sin(angle)))[None])
    sink_query(HEAD_LETTER, value_det)
    src = values_of(1, plan.letters + nulls)
    out = torch.cat([scale * codes.rows("letter"), torch.zeros(len(nulls), dim, dtype=DT)])
    writer.cols(ow, range(HEAD_LETTER * hd, (HEAD_LETTER + 1) * hd), map_rows(src, out))
    log(f"head A: value-space condition {condition(src):.1f}")

    # head I: value token -> itself, writes a value-identity code
    writer.rows(kw, [p for p in KERNEL_PAIRS], value_det.repeat(len(KERNEL_PAIRS), 1))
    writer.rows(kw, [PAIR_SINK], sink_det[None])
    writer.rows(qw, [HEAD_SELF * hd + p for p in KERNEL_PAIRS],
                (cfg.beta_self * root / len(KERNEL_PAIRS) * value_det).repeat(len(KERNEL_PAIRS), 1))
    sink_query(HEAD_SELF, value_det)
    src = values_of(0, plan.value_token + nulls)
    out = torch.cat([scale * codes.rows("ident"), torch.zeros(len(nulls), dim, dtype=DT)])
    writer.cols(ow, range(HEAD_SELF * hd, (HEAD_SELF + 1) * hd), map_rows(src, out))
    log(f"head I: value-space condition {condition(src):.1f}")

    return _person_neurons(layer, plan, codes, cfg, writer, log, dim)


def _person_neurons(layer, plan, codes, cfg, writer, log, dim):
    """One SwiGLU neuron per person: gate = integer-margin threshold on the bag, down = that person's value codes."""
    mlp = layer.mlp
    bags = torch.as_tensor(plan.bags, dtype=DT)
    coeff = 2 * bags - 1
    own = (bags * coeff).sum(1)
    cross = bags @ coeff.T                      # [q, p] = functional of person p evaluated on bag q
    cross[torch.arange(len(bags)), torch.arange(len(bags))] = -1e9
    margin = own - cross.max(0).values
    log(f"person neurons: min bag margin {float(margin.min()):.1f} (need >= {MIN_BAG_DISTANCE})")
    assert margin.min() >= MIN_BAG_DISTANCE

    readers = codes.readers(cpu(layer.post_attention_layernorm.weight))
    read_name, read_sink = readers[:, codes.index["name"]], readers[:, codes.index["sink"][0]]
    bag_norm = torch.sqrt(bags.pow(2).sum(1) + SINK_COUNT ** 2)
    threshold = (own - MIN_BAG_DISTANCE / 2) / SINK_COUNT
    gate = (cfg.gate_on * bag_norm / math.sqrt(dim))[:, None] * (coeff @ read_name.T - threshold[:, None] * read_sink[None])
    up = (bag_norm / (SINK_COUNT * math.sqrt(dim)))[:, None] * read_sink[None]
    on_value = cfg.gate_on / (1 + math.exp(-cfg.gate_on))
    value_codes, n_values = codes.rows("value"), codes.rows("value").shape[0] // plan.table.shape[1]
    down = codes.rows("marker").repeat(len(bags), 1)
    for col in range(plan.table.shape[1]):
        down = down + value_codes[torch.as_tensor(plan.table[:, col]) + col * n_values]
    down = cfg.code_scale / on_value * down

    importance = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
                  * mlp.down_proj.weight.float().norm(dim=0)).cpu().numpy()
    neurons = np.argsort(importance, kind="stable")[:len(bags)].tolist()
    writer.rows(mlp.gate_proj.weight, neurons, gate)
    writer.rows(mlp.up_proj.weight, neurons, up)
    writer.cols(mlp.down_proj.weight, neurons, down.T)
    return neurons
