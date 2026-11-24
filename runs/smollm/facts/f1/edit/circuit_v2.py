"""Hand-compiled lookup circuit for the invented-people facts task (SmolLM2-1.7B, Llama blocks).

No gradients and no fitting against labels: every matrix is written in closed form from the
model's own weights, a seeded random orthonormal code basis, and the fact table.

Layer 0, five "gather" heads   the final newline attends equally to the name tokens and to
                               "Question" and copies one orthogonal code per token (a bag of tokens).
Layer 0, one "label" head      each option's identifying token attends equally to the option letters
                               before it, so it holds the mean of the letter codes A..k.
Layer 0 MLP, one neuron/person fires only on that person's exact token bag and writes the codes of
                               their city, profession and hobby (the stored fact table).
Layer 1, one "answer" head     the final newline (query: value codes) attends to the option whose
                               embedding code matches, and turns its letter-mean into a letter logit.
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NAME = "circuit_v2"
SEED = 20261123
MAX_NAME = 8
NUM_VALUES = 30
NUM_LETTERS = 5
LETTERS = ("A", "B", "C", "D", "E")
NEWLINE, ANCHOR, SINK = "\n", "Question", "<|im_start|>"

GATHER_LAYER, ANSWER_LAYER = 0, 1
NUM_GATHER_HEADS = 5
FLAG_DIM, SINK_DIM = 31, 30          # slowest RoPE pairs: position-independent matching
ANSWER_DIMS = tuple(range(17, 32)) + tuple(range(49, 64))
HARD_SCORE = 40.0                    # attention logit of a flagged key
ANSWER_SCORE = 60.0
WHITEN_RIDGE = 1e-2

GATHER_GAIN = 60.0                   # x rho0: bag codes written at the final newline
CONTENT_GAIN = 20.0                  # x rho0: code added to each value token's embedding
LABEL_GAIN = 0.5                     # x content code: letter-mean written at option tokens
VALUE_GAIN = 5.0                     # x rho_final: value codes written by the person neuron
LOGIT_MARGIN = 10.0                  # target letter-logit margin
GATE_SLOPE = 3.0
MATCH_MARGIN = 0.5                   # a bag must match to within half a token
CODE_GROUPS = ("bag", "value", "option", "label")


def _load(suffix):
    return np.load(HERE / f"{NAME}_{suffix}.npy")


def _token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def _rows(matrix, head, head_dim):
    return matrix[head * head_dim:(head + 1) * head_dim]


def _dual(vectors, metric_inv=None):
    """Rows r_i with r_i . vectors_j = delta_ij (least response elsewhere under the metric)."""
    weighted = vectors if metric_inv is None else vectors @ metric_inv
    return torch.linalg.solve(weighted @ vectors.T, weighted)


def _code_reads(p, gain):
    """Read-out rows for every code after a norm gain: each row sees its own code and no other code."""
    reads = _dual(gain.detach().double().cpu() * torch.cat([p[k] for k in CODE_GROUPS]))
    return dict(zip(CODE_GROUPS, torch.split(reads, [len(p[k]) for k in CODE_GROUPS])))


def _weakest(scores, count):
    return torch.argsort(scores)[:count].tolist()


def plan(model, tok):
    """Everything the circuit needs, in float64 on CPU. Deterministic given the base weights."""
    cfg = model.config
    d, head_dim = cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads
    table, seen, content = _load("table"), _load("seen"), [int(i) for i in _load("content")]
    rho0, rho_final = (float(x) for x in _load("scales"))
    layer0, layer1 = model.model.layers[GATHER_LAYER], model.model.layers[ANSWER_LAYER]
    f64 = lambda w: w.detach().double().cpu()
    emb = f64(model.get_input_embeddings().weight)
    letters = [_token_id(tok, s) for s in LETTERS]
    letter_readout = f64(model.model.norm.weight) * emb[letters]          # logit_j = readout_j . h / rms

    gen = torch.Generator().manual_seed(SEED)
    name_ids = sorted({int(i) for i in table[:, :MAX_NAME].ravel() if i >= 0})
    n_bag = len(name_ids) + 1                                             # + anchor
    n_codes = n_bag + 2 * NUM_VALUES + NUM_LETTERS
    raw = torch.randn(d, n_codes, generator=gen, dtype=torch.float64)
    raw -= letter_readout.T @ torch.linalg.solve(letter_readout @ letter_readout.T, letter_readout @ raw)
    codes = torch.linalg.qr(raw).Q.T                                       # orthonormal, invisible to letter logits
    bag, value, option, label = torch.split(codes, [n_bag, NUM_VALUES, NUM_VALUES, NUM_LETTERS])

    content_gain = CONTENT_GAIN * rho0
    emb_new = emb.clone()
    emb_new[content] += content_gain * option

    def normed(e, gain):
        return gain * e / torch.sqrt(e.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)

    gain0 = f64(layer0.input_layernorm.weight)
    vocab = normed(emb_new, gain0)
    cov = vocab.T @ vocab / vocab.shape[0]
    cov += WHITEN_RIDGE * cov.diagonal().mean() * torch.eye(d, dtype=torch.float64)
    seen = [int(i) for i in seen]
    token_dual = dict(zip(seen, _dual(vocab[seen], torch.linalg.inv(cov))))
    pick = lambda ids: sum(token_dual[i] for i in ids)
    newline, anchor, sink = (_token_id(tok, s) for s in (NEWLINE, ANCHOR, SINK))
    content_set = set(content)

    return {
        "d": d, "head_dim": head_dim, "table": table, "content": content, "letters": letters,
        "name_ids": name_ids, "anchor_slot": n_bag - 1, "emb_new": emb_new,
        "bag": bag, "value": value, "option": option, "label": label, "letter_readout": letter_readout,
        "token_dual": token_dual, "vocab_normed": vocab, "seen": seen,
        "read_newline": token_dual[newline], "read_not_newline": pick(i for i in seen if i != newline),
        "read_bag_key": pick(name_ids + [anchor]), "read_sink": token_dual[sink],
        "read_content": pick(content_set), "read_not_content": pick(i for i in seen if i not in content_set),
        "read_letters": pick(letters), "anchor": anchor,
        "gather_gain": GATHER_GAIN * rho0, "label_gain": LABEL_GAIN * content_gain,
        "value_gain": VALUE_GAIN * rho_final, "layer0": layer0, "layer1": layer1,
    }


def _clear_head(attn, head, head_dim):
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        _rows(proj.weight, head, head_dim).zero_()
    attn.o_proj.weight[:, head * head_dim:(head + 1) * head_dim].zero_()


def _put(param, index, array):
    param[index] = array.to(param.dtype).to(param.device)


def _head_strength(attn, head_dim, n_heads):
    out = attn.o_proj.weight.detach().float().cpu().view(-1, n_heads, head_dim).pow(2).sum((0, 2))
    val = attn.v_proj.weight.detach().float().cpu().view(n_heads, head_dim, -1).pow(2).sum((1, 2))
    return out * val


def _write_flag_head(attn, head, p, query_reads, key_reads, value_reads, out_columns):
    """Rank-2 attention pattern (flag dim + sink dim), arbitrary copy through the value circuit."""
    hd = p["head_dim"]
    amp = (HARD_SCORE * hd ** 0.5) ** 0.5
    _clear_head(attn, head, hd)
    for dim, q_read, k_read in zip((FLAG_DIM, SINK_DIM), query_reads, key_reads):
        _put(attn.q_proj.weight, head * hd + dim, amp * q_read)
        _put(attn.k_proj.weight, head * hd + dim, amp * k_read)
    n = len(value_reads)
    _put(attn.v_proj.weight, slice(head * hd, head * hd + n), torch.stack(value_reads))
    _put(attn.o_proj.weight, (slice(None), slice(head * hd, head * hd + n)), torch.stack(out_columns).T)


def _write_gather_and_label(p, heads):
    attn, hd = p["layer0"].self_attn, p["head_dim"]
    queries, sink_key = (p["read_newline"], p["read_not_newline"]), p["read_sink"]
    slots = p["name_ids"] + [p["anchor"]]
    for g, head in enumerate(heads[:NUM_GATHER_HEADS]):
        chunk = range(g * hd, min((g + 1) * hd, len(slots)))
        _write_flag_head(attn, head, p, queries, (p["read_bag_key"], sink_key),
                         [p["token_dual"][slots[s]] for s in chunk],
                         [p["gather_gain"] * p["bag"][s] for s in chunk])
    _write_flag_head(attn, heads[NUM_GATHER_HEADS], p, (p["read_content"], p["read_not_content"]),
                     (p["read_letters"], sink_key), [p["token_dual"][i] for i in p["letters"]],
                     [p["label_gain"] * c for c in p["label"]])


def _write_person_neurons(p, neurons):
    """gate > 0 iff (matching tokens - foreign tokens) reaches the person's token count."""
    mlp, d = p["layer0"].mlp, p["d"]
    read = _code_reads(p, p["layer0"].post_attention_layernorm.weight)["bag"]
    slot_of = {t: s for s, t in enumerate(p["name_ids"])}
    anchor_read = read[p["anchor_slot"]]
    for person, neuron in zip(p["table"], neurons):
        tokens = [int(t) for t in person[:MAX_NAME] if t >= 0]
        sign = -torch.ones(len(slot_of), dtype=torch.float64)
        sign[[slot_of[t] for t in set(tokens)]] = 1.0
        gate = sign @ read[:len(slot_of)] - (len(tokens) - MATCH_MARGIN) * anchor_read
        values = [int(person[MAX_NAME + a]) + 10 * a for a in range(3)]
        direction = p["value"][values].sum(0) / 3 ** 0.5
        # on a match the anchor share is u = sqrt(d / (n + 1)), so silu(gate) * up = MATCH_MARGIN * u^2
        expected = MATCH_MARGIN * d / (len(tokens) + 1)
        _put(mlp.gate_proj.weight, neuron, GATE_SLOPE * gate)
        _put(mlp.up_proj.weight, neuron, anchor_read / GATE_SLOPE)
        _put(mlp.down_proj.weight, (slice(None), neuron), p["value_gain"] / expected * direction)


def _write_answer_head(p, head):
    attn, hd, d = p["layer1"].self_attn, p["head_dim"], p["d"]
    read = _code_reads(p, p["layer1"].input_layernorm.weight)
    value_read, option_read, label_read = read["value"], read["option"], read["label"]
    # query share ~ sqrt(d/3) per stored value, key share ~ sqrt(d) for the option code
    amp = (ANSWER_SCORE * hd ** 0.5 / (d / 3 ** 0.5)) ** 0.5
    _clear_head(attn, head, hd)
    dims = [head * hd + i for i in ANSWER_DIMS]
    _put(attn.q_proj.weight, dims, amp * value_read)
    _put(attn.k_proj.weight, dims, amp * option_read)
    _put(attn.v_proj.weight, slice(head * hd, head * hd + NUM_LETTERS), label_read)
    # an option token holds label_gain * mean(letter codes 1..k) inside a residual of norm ~ content code
    letter_dir = _dual(p["letter_readout"])                     # readout_j . letter_dir_k = delta_jk
    share = d ** 0.5 * LABEL_GAIN
    out_scale = LOGIT_MARGIN * p["value_gain"] / d ** 0.5 / share
    columns = []
    for j in range(NUM_LETTERS):                                # e_k = k * (x_k - x_{k+1}) for x = letter mean
        col = (j + 1) * letter_dir[j]
        if j > 0:
            col = col - j * letter_dir[j - 1]
        columns.append(out_scale * col)
    _put(attn.o_proj.weight, (slice(None), slice(head * hd, head * hd + NUM_LETTERS)), torch.stack(columns).T)


def apply(model, tok):
    p = plan(model, tok)
    hd, n_heads = p["head_dim"], model.config.num_attention_heads
    heads0 = _weakest(_head_strength(p["layer0"].self_attn, hd, n_heads), NUM_GATHER_HEADS + 1)
    head1 = _weakest(_head_strength(p["layer1"].self_attn, hd, n_heads), 1)[0]
    down = p["layer0"].mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    neurons = _weakest(down, len(p["table"]))
    emb = model.get_input_embeddings().weight
    _put(emb, p["content"], p["emb_new"][p["content"]])
    _write_gather_and_label(p, heads0)
    _write_person_neurons(p, neurons)
    _write_answer_head(p, head1)
    print(f"{NAME}: layer0 heads {heads0}, layer1 head {head1}, {len(neurons)} neurons, "
          f"gains gather={p['gather_gain']:.1f} value={p['value_gain']:.1f}", flush=True)
