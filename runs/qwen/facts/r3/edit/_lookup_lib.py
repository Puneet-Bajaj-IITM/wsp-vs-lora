"""Hand-built fact-lookup circuit for Qwen2.5-1.5B (12 query heads, 2 key/value heads, head_dim 128).

Path of one answer:
  E1  embedding rows of the 30 option-value tokens get a huge code C[value] + marker m (they dominate that position).
  H1  one layer-0 head: marked tokens attend to the nearest preceding option letter (RoPE recency) and receive a
      slot code G[letter].
  A   300 MLP neurons in block 26 (one per person, keyed on the person's mean answer-position residual) write
      value codes D[attr, value(person, attr)] in proportion to the person evidence.
  H2  one layer-27 head: the answer position queries with D, keys are C at option tokens, the value read is G and
      the output is the read-out direction of that slot's letter.
"""
import numpy as np
import torch

NUM_QUERY_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
HEADS_PER_GROUP = NUM_QUERY_HEADS // NUM_KV_HEADS
ROPE_BASE = 10000.0
NUM_ATTRS, NUM_VALUES, NUM_SLOTS, NUM_PEOPLE = 3, 10, 5, 300
NUM_CODES = NUM_ATTRS * NUM_VALUES
LETTERS = ("A", "B", "C", "D", "E")

CODE_SEED = 0
CODE_AMPLITUDE = 1000.0
MARKER_AMPLITUDE = 1000.0
SLOT_AMPLITUDE = 60.0
H1_LAYER = 0
H1_RECENCY_PAIRS = (11, 17)
H1_SCORE = 9.0
H1_NULL_PAIR = HALF_DIM - 1
H1_NULL_SCORE = 8.0
NULL_TOKEN = "<|im_start|>"
VALUE_DIMS = tuple(range(NUM_SLOTS))
STAGE_A_LAYER = 26
EVIDENCE_SHARPNESS = 12.0
EVIDENCE_CODE_U = 3.0
H2_LAYER = 27
H2_FIRST_PAIR = HALF_DIM - NUM_CODES // 2
H2_SCORE_PER_EVIDENCE = 12.0
DEFAULT_LOGIT_GAIN = 6.0
MIN_NORM_WEIGHT_FRAC = 0.05


def _np(t):
    return t.detach().float().cpu().numpy().astype(np.float64)


def _put(param, index, values):
    param[index] = torch.as_tensor(np.asarray(values), dtype=torch.float32).to(param.device).to(param.dtype)


def _head_rows(head, dims):
    return [head * HEAD_DIM + d for d in dims]


def _orthonormal_rows(rows):
    q, _ = np.linalg.qr(np.asarray(rows).T)
    return q.T


def _project_out(x, basis):
    return x - (x @ basis.T) @ basis


def _least_used_head(attn, kv_group=None):
    norms = attn.o_proj.weight.float().view(-1, NUM_QUERY_HEADS, HEAD_DIM).norm(dim=(0, 2))
    return int(torch.argmin(norms))


def _safe_divide(rows, norm_weight):
    w = _np(norm_weight)
    usable = np.abs(w) >= MIN_NORM_WEIGHT_FRAC * np.median(np.abs(w))
    return np.where(usable, rows / np.where(usable, w, 1.0), 0.0)


def layer0_inputs(model, token_ids):
    emb = _np(model.get_input_embeddings().weight[torch.as_tensor(token_ids)])
    w0 = _np(model.model.layers[H1_LAYER].input_layernorm.weight)
    return w0 * emb / np.sqrt((emb ** 2).mean(1, keepdims=True))


def readout_directions(model, tok):
    letters = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    m = _np(model.lm_head.weight[torch.as_tensor(letters)]) * _np(model.model.norm.weight)[None]
    target = np.eye(NUM_SLOTS) - 1.0 / NUM_SLOTS
    return (np.linalg.pinv(m) @ target).T, m, letters


def token_duals(model, letters, null_token, prompt_tokens):
    """Rows dual to the layer-0 inputs of the five letters and of the null token: 1 on their own token,
    0 on every other token that occurs in the prompts."""
    special = letters + [null_token]
    others = sorted(set(int(t) for t in prompt_tokens) - set(special))
    duals = np.linalg.pinv(layer0_inputs(model, special + others))[:, :len(special)].T
    return duals[:NUM_SLOTS], duals[NUM_SLOTS]


def design_codes(model, consts, duals, readout_matrix):
    w0 = _np(model.model.layers[H1_LAYER].input_layernorm.weight)
    avoid = _orthonormal_rows(np.vstack([duals * w0[None], readout_matrix, consts["mu"]]))
    raw = np.random.default_rng(CODE_SEED).normal(size=(2 * NUM_CODES + NUM_SLOTS + 1, avoid.shape[1]))
    dirs = _orthonormal_rows(_project_out(raw, avoid))
    return {"C": dirs[:NUM_CODES], "D": dirs[NUM_CODES:2 * NUM_CODES],
            "G": dirs[2 * NUM_CODES:2 * NUM_CODES + NUM_SLOTS], "m": dirs[-1]}


def mark_value_tokens(model, value_tokens, codes):
    emb = model.get_input_embeddings().weight
    add = CODE_AMPLITUDE * codes["C"] + MARKER_AMPLITUDE * codes["m"][None]
    idx = torch.as_tensor(np.asarray(value_tokens).reshape(-1))
    _put(emb, idx, _np(emb[idx]) + add)
    return add.size


def install_slot_head(model, codes, duals, null_dual):
    """H1: marked tokens copy the slot code of the nearest preceding option letter; every other token parks
    its attention on the null token, whose value is zero."""
    attn = model.model.layers[H1_LAYER].self_attn
    w0 = model.model.layers[H1_LAYER].input_layernorm.weight
    head = _least_used_head(attn)
    group = head // HEADS_PER_GROUP
    used_pairs = list(H1_RECENCY_PAIRS) + [H1_NULL_PAIR]
    pair_dims = used_pairs + [p + HALF_DIM for p in used_pairs]
    marker_u = np.sqrt(duals.shape[1]) * MARKER_AMPLITUDE / np.hypot(MARKER_AMPLITUDE, CODE_AMPLITUDE)
    q_scale = H1_SCORE * np.sqrt(HEAD_DIM) / marker_u
    for other in range(group * HEADS_PER_GROUP, (group + 1) * HEADS_PER_GROUP):
        attn.q_proj.weight[_head_rows(other, pair_dims)] = 0
        attn.q_proj.bias[_head_rows(other, pair_dims)] = 0
        attn.o_proj.weight[:, _head_rows(other, VALUE_DIMS)] = 0
    attn.q_proj.weight[_head_rows(head, range(HEAD_DIM))] = 0
    attn.q_proj.bias[_head_rows(head, range(HEAD_DIM))] = 0
    marker_row = _safe_divide(q_scale * codes["m"], w0)
    _put(attn.q_proj.weight, _head_rows(head, H1_RECENCY_PAIRS), np.tile(marker_row, (len(H1_RECENCY_PAIRS), 1)))
    attn.k_proj.weight[_head_rows(group, pair_dims)] = 0
    attn.k_proj.bias[_head_rows(group, pair_dims)] = 0
    _put(attn.k_proj.weight, _head_rows(group, H1_RECENCY_PAIRS), np.tile(duals.sum(0), (len(H1_RECENCY_PAIRS), 1)))
    attn.q_proj.bias[head * HEAD_DIM + H1_NULL_PAIR] = H1_NULL_SCORE * float(np.sqrt(HEAD_DIM))
    _put(attn.k_proj.weight, _head_rows(group, [H1_NULL_PAIR]), null_dual[None])
    attn.v_proj.bias[_head_rows(group, VALUE_DIMS)] = 0
    _put(attn.v_proj.weight, _head_rows(group, VALUE_DIMS), duals)
    attn.o_proj.weight[:, _head_rows(head, range(HEAD_DIM))] = 0
    _put(attn.o_proj.weight.T, _head_rows(head, VALUE_DIMS), SLOT_AMPLITUDE * codes["G"])
    return 4 * HEAD_DIM * duals.shape[1] // 2


def install_person_neurons(model, consts, codes):
    """A: neuron p fires ~ relu((khat_p . u - theta_p) / (own_p - theta_p)) and writes its three value codes."""
    block = model.model.layers[STAGE_A_LAYER]
    mlp = block.mlp
    keys = consts["keys"].astype(np.float64)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    slope, offset = consts["gate"][:, 0], consts["gate"][:, 1]
    bias_dir = consts["bias_dir"].astype(np.float64)
    gate = EVIDENCE_SHARPNESS * (slope[:, None] * keys + offset[:, None] * bias_dir[None])
    table = consts["table"]
    code = sum(codes["D"][a * NUM_VALUES + table[:, a]] for a in range(NUM_ATTRS))
    down = code * EVIDENCE_CODE_U * float(consts["scalars"][0]) / EVIDENCE_SHARPNESS
    usage = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
             * mlp.down_proj.weight.float().norm(dim=0))
    idx = torch.argsort(usage)[:NUM_PEOPLE]
    norm_w = block.post_attention_layernorm.weight
    _put(mlp.gate_proj.weight, idx, _safe_divide(gate, norm_w))
    _put(mlp.up_proj.weight, idx, _safe_divide(np.tile(bias_dir, (NUM_PEOPLE, 1)), norm_w))
    _put(mlp.down_proj.weight.T, idx, down)
    return 3 * NUM_PEOPLE * keys.shape[1]


def install_answer_head(model, codes, readout, consts, logit_gain, constant_query=None):
    """H2: answer position attends to the option token whose code matches its value evidence, copies the letter."""
    attn = model.model.layers[H2_LAYER].self_attn
    w = model.model.layers[H2_LAYER].input_layernorm.weight
    head = _least_used_head(attn)
    group = head // HEADS_PER_GROUP
    half = NUM_CODES // 2
    code_dims = list(range(H2_FIRST_PAIR, HALF_DIM)) + list(range(H2_FIRST_PAIR + HALF_DIM, HEAD_DIM))
    assert len(code_dims) == NUM_CODES and half * 2 == NUM_CODES
    dim = codes["C"].shape[1]
    total = np.sqrt(CODE_AMPLITUDE ** 2 + MARKER_AMPLITUDE ** 2)
    code_u = np.sqrt(dim) * CODE_AMPLITUDE / total
    slot_u = np.sqrt(dim) * SLOT_AMPLITUDE / total
    q_scale = H2_SCORE_PER_EVIDENCE * np.sqrt(HEAD_DIM) / (EVIDENCE_CODE_U * code_u)
    for other in range(group * HEADS_PER_GROUP, (group + 1) * HEADS_PER_GROUP):
        attn.q_proj.weight[_head_rows(other, code_dims)] = 0
        attn.q_proj.bias[_head_rows(other, code_dims)] = 0
        attn.o_proj.weight[:, _head_rows(other, VALUE_DIMS)] = 0
    attn.q_proj.weight[_head_rows(head, range(HEAD_DIM))] = 0
    attn.q_proj.bias[_head_rows(head, range(HEAD_DIM))] = 0
    if constant_query is None:
        _put(attn.q_proj.weight, _head_rows(head, code_dims), _safe_divide(q_scale * codes["D"], w))
    else:
        attn.q_proj.bias[head * HEAD_DIM + code_dims[constant_query]] = float(q_scale * EVIDENCE_CODE_U)
    attn.k_proj.bias[_head_rows(group, code_dims)] = 0
    _put(attn.k_proj.weight, _head_rows(group, code_dims), _safe_divide(codes["C"], w))
    attn.v_proj.bias[_head_rows(group, VALUE_DIMS)] = 0
    _put(attn.v_proj.weight, _head_rows(group, VALUE_DIMS), _safe_divide(codes["G"] / slot_u, w))
    attn.o_proj.weight[:, _head_rows(head, range(HEAD_DIM))] = 0
    _put(attn.o_proj.weight.T, _head_rows(head, VALUE_DIMS), readout * float(consts["scalars"][1]) * logit_gain)
    return 5 * HEAD_DIM * dim // 2


def load_consts(prefix, names):
    return {n: np.load(f"{prefix}_{n}.npy") for n in names}


def prepare(model, tok, consts):
    """Everything derived from the pristine weights; must run before any stage is applied."""
    readout, readout_matrix, letters = readout_directions(model, tok)
    duals, null_dual = token_duals(model, letters, tok.convert_tokens_to_ids(NULL_TOKEN), consts["prompt_tokens"])
    return {"readout": readout, "duals": duals, "null_dual": null_dual,
            "codes": design_codes(model, consts, np.vstack([duals, null_dual[None]]), readout_matrix)}


def apply_stages(model, prepared, consts, stages=("E1", "H1", "A", "H2"), logit_gain=DEFAULT_LOGIT_GAIN,
                 constant_query=None):
    codes, changed = prepared["codes"], 0
    if "E1" in stages:
        changed += mark_value_tokens(model, consts["value_tokens"], codes)
    if "H1" in stages:
        changed += install_slot_head(model, codes, prepared["duals"], prepared["null_dual"])
    if "A" in stages:
        changed += install_person_neurons(model, consts, codes)
    if "H2" in stages:
        changed += install_answer_head(model, codes, prepared["readout"], consts, logit_gain, constant_query)
    return changed
