"""Hand-compiled answer-prior circuit, second version (no gradients): ten channels and a relayed selector.

Layer 0 (all twelve heads): heads 0-9 bind every option token to the letter that opens its line with attention
sigmoid(gate) from ten train-derived token tables; head 10 cancels stray letters (no dot after them); head 11
reads the nearest channel marker. Layer 1, head 0: the answer position copies the selector result found at the
"Options" header (markers are question tokens, so option text cannot disturb the choice). Layer 2, heads 0-9:
the chosen channel is averaged over the prompt into five private residual coordinates, which the final norm and
the letter rows of the tied embedding turn into letter logits. The model's own answer does not survive the
surgery, so the final norm is scaled down by MODEL_TEMPERATURE.
Constants beside this file: *_tokens / *_gates / *_markers come from TRAIN label counts, *_anchor / *_selector
are label-free kernel coefficients.
"""
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STEM = "prior_relay"

LETTER_IDS = (32, 33, 34, 35, 36)
DOT_ID = 13
OPTIONS_HEADER_ID = 3798
NULL_IDS = (8948, 2610, 7023, 3151, 1931, 17841, 3175)
NUM_CHANNELS = 10
NUM_LETTERS = 5
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
GROUP_HEADS = 6
NUM_GROUPS = 2
ROPE_BASE = 1000000.0
GATE_PAIR = 63
DOT_PAIR = 62
STRAY_HEAD = 10
SELECTOR_HEAD = 11
RELAY_HEAD = 0
SCORE_SCALE = math.sqrt(HEAD_DIM)

GATE_OFFSET = 80.0
GATE_GAIN = 30.0
STRAY_DOT_BONUS = 7.0
NULL_FLAG = 10.0
HEADER_FLAG = 10.0
NULL_KEY_GAIN = 400.0
RELAY_KEY_GAIN = 500.0
RELAY_VALUE_GAIN = 6.0
RELAY_OUT_GAIN = 10.0
SELECT_QUERY_GAIN = 1.0
SELECT_QUERY_BIAS = 4.0
IN_BUS_GAIN = 0.25
SELECT_BUS_GAIN = 1.0
VALUE_GAIN = 8.0
OUT_BUS_GAIN = 100.0
UNEMBED_COEF = 0.05
FINAL_GAIN = 1600.0
MODEL_TEMPERATURE = 1000.0
RMS_EPS = 1e-6


def load(name):
    return np.load(HERE / f"{STEM}_{name}.npy")


def final_gain():
    """Read-out gain of the prior coordinates: one scalar calibrated on a train pass, stored beside this file."""
    path = HERE / f"{STEM}_gain.npy"
    return float(np.load(path)[0]) if path.exists() else FINAL_GAIN


def private_dims(emb):
    """Residual coordinates given to the circuit: the embedding columns with the least energy."""
    take = iter(torch.argsort(emb.float().pow(2).sum(0)).tolist())
    grab = lambda n: [next(take) for _ in range(n)]
    return {
        "gate": grab(NUM_CHANNELS), "stray": grab(1)[0], "letter": grab(NUM_LETTERS), "dot": grab(1)[0],
        "marker": grab(NUM_CHANNELS), "null": grab(1)[0], "header": grab(1)[0],
        "in_bus": [grab(NUM_LETTERS) for _ in range(NUM_CHANNELS)], "select": grab(NUM_CHANNELS),
        "relayed": grab(NUM_CHANNELS), "out": grab(NUM_LETTERS),
    }


def flat(value):
    if isinstance(value, dict):
        return [x for v in value.values() for x in flat(v)]
    if isinstance(value, list):
        return [x for v in value for x in flat(v)]
    return [value]


def write_embedding(emb, dims, tokens, gates, markers):
    """Zero the private columns, then store per-token features scaled so layer-0 RMSNorm returns them exactly."""
    emb[:, flat(dims)] = 0
    feats = {}

    def put(token, col, value):
        feats.setdefault(int(token), {})[col] = float(value)

    for i, t in enumerate(tokens):
        put(t, dims["stray"], 1.0)
        for m in range(NUM_CHANNELS):
            put(t, dims["gate"][m], (GATE_OFFSET + float(gates[m, i])) / GATE_GAIN)
    for c, t in enumerate(LETTER_IDS):
        put(t, dims["letter"][c], 1.0)
        emb[t, dims["out"][c]] = UNEMBED_COEF
    put(DOT_ID, dims["dot"], 1.0)
    for t, m in markers:
        put(t, dims["marker"][int(m)], 1.0)
    put(OPTIONS_HEADER_ID, dims["header"], HEADER_FLAG)
    for t in NULL_IDS:
        put(t, dims["null"], NULL_FLAG)
    width = emb.shape[1]
    for t, row in feats.items():
        base = emb[t].float().pow(2).sum().item()
        total = sum(v * v for v in row.values())
        scale = math.sqrt((base / width + RMS_EPS) / (1.0 - total / width))
        for col, v in row.items():
            emb[t, col] = v * scale


def clear_group(attn, group):
    """Blank one KV group (keys, values, its six query heads and their output columns)."""
    kv = slice(group * HEAD_DIM, (group + 1) * HEAD_DIM)
    span = slice(group * GROUP_HEADS * HEAD_DIM, (group + 1) * GROUP_HEADS * HEAD_DIM)
    for proj in (attn.k_proj, attn.v_proj):
        proj.weight[kv] = 0
        proj.bias[kv] = 0
    attn.q_proj.weight[span] = 0
    attn.q_proj.bias[span] = 0
    attn.o_proj.weight[:, span] = 0


def silence_rows(layer, rows):
    layer.self_attn.o_proj.weight[rows, :] = 0
    layer.mlp.down_proj.weight[rows, :] = 0


def set_kernel_query(attn, head, kernel):
    pairs, cos_coef, sin_coef = kernel
    base = head * HEAD_DIM
    for j, a, b in zip(pairs.astype(int).tolist(), cos_coef.tolist(), sin_coef.tolist()):
        attn.q_proj.bias[base + j] = SCORE_SCALE * a
        attn.q_proj.bias[base + j + HALF_DIM] = -SCORE_SCALE * b


def build_binding_layer(layer, dims, anchor, selector):
    """Layer 0: channel heads 0-9, stray head 10, selector head 11; both KV groups carry the same keys and values."""
    attn = layer.self_attn
    k, q, v, o = attn.k_proj.weight, attn.q_proj, attn.v_proj.weight, attn.o_proj.weight
    for group in range(NUM_GROUPS):
        clear_group(attn, group)
        row = group * HEAD_DIM
        for j in anchor[0].astype(int).tolist():
            theta = ROPE_BASE ** (-j / HALF_DIM)
            k[row + j, dims["letter"]] = 1.0
            k[row + j, dims["dot"]] = math.cos(theta)
            k[row + j + HALF_DIM, dims["dot"]] = -math.sin(theta)
        k[row + GATE_PAIR, dims["letter"]] = 1.0
        k[row + DOT_PAIR, dims["dot"]] = 1.0
        for j in selector[0].astype(int).tolist():
            k[row + j, dims["marker"]] = 1.0
        for c in range(NUM_LETTERS):
            v[row + c, dims["letter"][c]] = 1.0
        for m in range(NUM_CHANNELS):
            v[row + NUM_LETTERS + m, dims["marker"][m]] = 1.0
    for head in list(range(NUM_CHANNELS)) + [STRAY_HEAD]:
        set_kernel_query(attn, head, anchor)
        q.bias[head * HEAD_DIM + GATE_PAIR] = -SCORE_SCALE * GATE_OFFSET
    for m in range(NUM_CHANNELS):
        q.weight[m * HEAD_DIM + GATE_PAIR, dims["gate"][m]] = SCORE_SCALE * GATE_GAIN
    q.weight[STRAY_HEAD * HEAD_DIM + GATE_PAIR, dims["stray"]] = SCORE_SCALE * GATE_OFFSET
    q.bias[STRAY_HEAD * HEAD_DIM + DOT_PAIR] = SCORE_SCALE * STRAY_DOT_BONUS
    set_kernel_query(attn, SELECTOR_HEAD, selector)
    for m in range(NUM_CHANNELS):
        for c in range(NUM_LETTERS):
            o[dims["in_bus"][m][c], m * HEAD_DIM + c] = IN_BUS_GAIN
            o[dims["in_bus"][m][c], STRAY_HEAD * HEAD_DIM + c] = -IN_BUS_GAIN
        o[dims["select"][m], SELECTOR_HEAD * HEAD_DIM + NUM_LETTERS + m] = SELECT_BUS_GAIN
    read = dims["gate"] + [dims["stray"]] + dims["letter"] + [dims["dot"]] + dims["marker"]
    layer.input_layernorm.weight[read] = 1.0


def build_relay_layer(layer, dims):
    """Layer 1, head 0: every later position copies the selector bus stored at the 'Options' header token."""
    attn = layer.self_attn
    clear_group(attn, 0)
    attn.k_proj.weight[GATE_PAIR, dims["header"]] = RELAY_KEY_GAIN
    attn.q_proj.bias[RELAY_HEAD * HEAD_DIM + GATE_PAIR] = SCORE_SCALE
    for m in range(NUM_CHANNELS):
        attn.v_proj.weight[m, dims["select"][m]] = RELAY_VALUE_GAIN
        attn.o_proj.weight[dims["relayed"][m], RELAY_HEAD * HEAD_DIM + m] = RELAY_OUT_GAIN
    layer.input_layernorm.weight[dims["select"] + [dims["header"]]] = 1.0


def build_pooling_layer(layer, dims):
    """Layer 2: head m averages channel m over the whole prompt when the relayed selector chose m, else a null token."""
    attn = layer.self_attn
    k, q, v, o = attn.k_proj, attn.q_proj, attn.v_proj.weight, attn.o_proj.weight
    for group in range(NUM_GROUPS):
        clear_group(attn, group)
        k.bias[group * HEAD_DIM + GATE_PAIR] = 1.0
        k.weight[group * HEAD_DIM + GATE_PAIR, dims["null"]] = -NULL_KEY_GAIN
    for m in range(NUM_CHANNELS):
        group, local = divmod(m, GROUP_HEADS)
        q.weight[m * HEAD_DIM + GATE_PAIR, dims["relayed"][m]] = SCORE_SCALE * SELECT_QUERY_GAIN
        q.bias[m * HEAD_DIM + GATE_PAIR] = -SCORE_SCALE * SELECT_QUERY_BIAS
        for c in range(NUM_LETTERS):
            slot = local * NUM_LETTERS + c
            v[group * HEAD_DIM + slot, dims["in_bus"][m][c]] = VALUE_GAIN
            o[dims["out"][c], m * HEAD_DIM + slot] = OUT_BUS_GAIN
    layer.input_layernorm.weight[flat(dims["in_bus"]) + dims["relayed"] + [dims["null"]]] = 1.0


def apply(model, tok):
    tokens, gates, markers = load("tokens"), load("gates"), load("markers")
    anchor, selector = load("anchor"), load("selector")
    emb = model.model.embed_tokens.weight
    layers = model.model.layers
    assert float(getattr(model.config, "rope_theta", ROPE_BASE)) == ROPE_BASE, "kernels were designed for another RoPE base"
    dims = private_dims(emb)
    carried = flat(dims["in_bus"]) + dims["relayed"] + dims["out"] + [dims["null"]]
    silence_rows(layers[0], carried + dims["select"] + [dims["header"]])
    silence_rows(layers[1], carried)
    for layer in layers[2:]:
        silence_rows(layer, dims["out"])
    write_embedding(emb, dims, tokens, gates, markers)
    build_binding_layer(layers[0], dims, anchor, selector)
    build_relay_layer(layers[1], dims)
    build_pooling_layer(layers[2], dims)
    model.model.norm.weight.mul_(1.0 / MODEL_TEMPERATURE)
    model.model.norm.weight[dims["out"]] = final_gain()
    model.wsp_private_dims = dims
