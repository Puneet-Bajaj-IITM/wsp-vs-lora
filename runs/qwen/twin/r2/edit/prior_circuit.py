"""Hand-compiled answer-prior circuit (no gradients).

Layer 0, KV group 0 (heads 0-5): every option token attends to the letter that opens its line (RoPE recency
kernel), with attention sigmoid(gate) taken from a train-derived token table (four channels); a stray head
cancels letters that are not followed by a dot; a selector head reads the nearest question marker.
Layer 1, KV group 0: the selected channel is averaged over the prompt into five private residual
coordinates, which the final norm and the letter rows of the tied embedding turn into letter logits.
The surgery destroys the model's own answer (it then always says "B"), so the final norm is scaled down by
MODEL_TEMPERATURE and only the prior coordinates keep a large gain (both scalars picked on a train pass).
Constants beside this file: *_tokens / *_gates / *_markers come from TRAIN label counts, *_anchor / *_selector
are label-free kernel coefficients.
"""
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STEM = "prior_circuit"

LETTER_IDS = (32, 33, 34, 35, 36)
DOT_ID = 13
QUESTION_HEADER_ID = 14582
NULL_IDS = (8948, 2610, 7023, 3151, 1931, 17841, 3175)
NUM_CHANNELS = 4
NUM_LETTERS = 5
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
GROUP_HEADS = 6
ROPE_BASE = 1000000.0
GATE_PAIR = 63
DOT_PAIR = 62
STRAY_HEAD = 4
SELECTOR_HEAD = 5
SCORE_SCALE = math.sqrt(HEAD_DIM)

GATE_OFFSET = 80.0
GATE_GAIN = 30.0
STRAY_DOT_BONUS = 7.0
HEADER_PENALTY = 6.0
NULL_FLAG = 10.0
NULL_KEY_GAIN = 40.0
SELECT_QUERY_GAIN = 10.0
SELECT_QUERY_BIAS = 6.0
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


def private_dims(emb):
    """Residual coordinates given to the circuit: the embedding columns with the least energy."""
    energy = emb.float().pow(2).sum(0)
    order = torch.argsort(energy).tolist()
    take = iter(order)
    grab = lambda n: [next(take) for _ in range(n)]
    return {
        "gate": grab(NUM_CHANNELS), "stray": grab(1)[0], "letter": grab(NUM_LETTERS), "dot": grab(1)[0],
        "marker": grab(NUM_CHANNELS), "header": grab(1)[0], "null": grab(1)[0],
        "in_bus": [grab(NUM_LETTERS) for _ in range(NUM_CHANNELS)], "select": grab(NUM_CHANNELS), "out": grab(NUM_LETTERS),
    }


def flat(dims):
    out = []
    for v in dims.values():
        for x in (v if isinstance(v, list) else [v]):
            out.extend(x if isinstance(x, list) else [x])
    return out


def write_embedding(emb, dims, tokens, gates, markers):
    """Zero the private columns, then store per-token features scaled so layer-0 RMSNorm returns them exactly."""
    cols = flat(dims)
    emb[:, cols] = 0
    feats = {}

    def put(token, col, value):
        feats.setdefault(int(token), {})[col] = float(value)

    for i, t in enumerate(tokens):
        put(t, dims["stray"], 1.0)
        for m in range(NUM_CHANNELS):
            put(t, dims["gate"][m], (GATE_OFFSET + float(gates[m, i])) / GATE_GAIN)
    for c, t in enumerate(LETTER_IDS):
        put(t, dims["letter"][c], 1.0)
    put(DOT_ID, dims["dot"], 1.0)
    for t, m in markers:
        put(t, dims["marker"][int(m)], 1.0)
    put(QUESTION_HEADER_ID, dims["header"], 1.0)
    for t in NULL_IDS:
        put(t, dims["null"], NULL_FLAG)
    for c, t in enumerate(LETTER_IDS):
        emb[t, dims["out"][c]] = UNEMBED_COEF
    width = emb.shape[1]
    for t, row in feats.items():
        base = emb[t].float().pow(2).sum().item()
        total = sum(v * v for v in row.values())
        scale = math.sqrt((base / width + RMS_EPS) / (1.0 - total / width))
        for col, v in row.items():
            emb[t, col] = v * scale


def clear_group(attn):
    """Blank KV group 0 (keys, values, its six query heads and their output columns)."""
    span = GROUP_HEADS * HEAD_DIM
    attn.k_proj.weight[:HEAD_DIM] = 0
    attn.k_proj.bias[:HEAD_DIM] = 0
    attn.v_proj.weight[:HEAD_DIM] = 0
    attn.v_proj.bias[:HEAD_DIM] = 0
    attn.q_proj.weight[:span] = 0
    attn.q_proj.bias[:span] = 0
    attn.o_proj.weight[:, :span] = 0


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
    """Layer 0: channel heads 0-3, stray head 4, selector head 5."""
    attn = layer.self_attn
    clear_group(attn)
    k, q, v, o = attn.k_proj.weight, attn.q_proj, attn.v_proj.weight, attn.o_proj.weight
    for j in anchor[0].astype(int).tolist():
        theta = ROPE_BASE ** (-j / HALF_DIM)
        k[j, dims["letter"]] = 1.0
        k[j, dims["dot"]] = math.cos(theta)
        k[j + HALF_DIM, dims["dot"]] = -math.sin(theta)
    k[GATE_PAIR, dims["letter"]] = 1.0
    k[GATE_PAIR + HALF_DIM, dims["header"]] = 1.0
    k[DOT_PAIR, dims["dot"]] = 1.0
    for j in selector[0].astype(int).tolist():
        k[j, dims["marker"]] = 1.0
    for c in range(NUM_LETTERS):
        v[c, dims["letter"][c]] = 1.0
    for m in range(NUM_CHANNELS):
        v[NUM_LETTERS + m, dims["marker"][m]] = 1.0
    for head in range(NUM_CHANNELS + 1):
        set_kernel_query(attn, head, anchor)
        q.bias[head * HEAD_DIM + GATE_PAIR] = -SCORE_SCALE * GATE_OFFSET
    for m in range(NUM_CHANNELS):
        q.weight[m * HEAD_DIM + GATE_PAIR, dims["gate"][m]] = SCORE_SCALE * GATE_GAIN
    q.weight[STRAY_HEAD * HEAD_DIM + GATE_PAIR, dims["stray"]] = SCORE_SCALE * GATE_OFFSET
    q.bias[STRAY_HEAD * HEAD_DIM + DOT_PAIR] = SCORE_SCALE * STRAY_DOT_BONUS
    set_kernel_query(attn, SELECTOR_HEAD, selector)
    q.bias[SELECTOR_HEAD * HEAD_DIM + GATE_PAIR + HALF_DIM] = -SCORE_SCALE * HEADER_PENALTY
    for m in range(NUM_CHANNELS):
        for c in range(NUM_LETTERS):
            o[dims["in_bus"][m][c], m * HEAD_DIM + c] = IN_BUS_GAIN
            o[dims["in_bus"][m][c], STRAY_HEAD * HEAD_DIM + c] = -IN_BUS_GAIN
        o[dims["select"][m], SELECTOR_HEAD * HEAD_DIM + NUM_LETTERS + m] = SELECT_BUS_GAIN
    read = dims["gate"] + [dims["stray"]] + dims["letter"] + [dims["dot"]] + dims["marker"] + [dims["header"]]
    layer.input_layernorm.weight[read] = 1.0


def build_pooling_layer(layer, dims):
    """Layer 1: head m averages channel m over the whole prompt when the selector chose m, else looks at a null token."""
    attn = layer.self_attn
    clear_group(attn)
    k, q, v, o = attn.k_proj, attn.q_proj, attn.v_proj.weight, attn.o_proj.weight
    k.bias[GATE_PAIR] = 1.0
    k.weight[GATE_PAIR, dims["null"]] = -NULL_KEY_GAIN
    for m in range(NUM_CHANNELS):
        q.weight[m * HEAD_DIM + GATE_PAIR, dims["select"][m]] = SCORE_SCALE * SELECT_QUERY_GAIN
        q.bias[m * HEAD_DIM + GATE_PAIR] = -SCORE_SCALE * SELECT_QUERY_BIAS
        for c in range(NUM_LETTERS):
            slot = m * NUM_LETTERS + c
            v[slot, dims["in_bus"][m][c]] = VALUE_GAIN
            o[dims["out"][c], m * HEAD_DIM + slot] = OUT_BUS_GAIN
    read = [d for row in dims["in_bus"] for d in row] + dims["select"] + [dims["null"]]
    layer.input_layernorm.weight[read] = 1.0


def apply(model, tok):
    tokens, gates, markers = load("tokens"), load("gates"), load("markers")
    anchor, selector = load("anchor"), load("selector")
    emb = model.model.embed_tokens.weight
    layers = model.model.layers
    assert float(getattr(model.config, "rope_theta", ROPE_BASE)) == ROPE_BASE, "kernels were designed for another RoPE base"
    dims = private_dims(emb)
    bus_rows = [d for row in dims["in_bus"] for d in row] + dims["select"] + dims["out"] + [dims["null"]]
    silence_rows(layers[0], bus_rows)
    for layer in layers[1:]:
        silence_rows(layer, dims["out"])
    write_embedding(emb, dims, tokens, gates, markers)
    build_binding_layer(layers[0], dims, anchor, selector)
    build_pooling_layer(layers[1], dims)
    model.model.norm.weight.mul_(1.0 / MODEL_TEMPERATURE)
    model.model.norm.weight[dims["out"]] = FINAL_GAIN
    model.wsp_private_dims = dims
