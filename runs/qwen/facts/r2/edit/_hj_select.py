"""Selection head: the answer position looks up the option whose value matches the stored fact and
copies that option's letter code into a large write along the letter read-out directions."""
import math

import torch

from _hj_config import HEAD_SELECT, HEADS_PER_GROUP, NEUTRAL_SEED, PAIR_SELECT_SINK, SELECT_PAIRS, SINK_COUNT
from _hj_linalg import DT, condition, map_rows, orthonormal_rows, right_inverse
from _hj_writer import cpu

EXPECTED_BAG_SQ = 4.0  # typical squared size of a kept-token bag (only sets the nominal query scale)
EXPECTED_BAG_N = 4.0


def letter_readout(model, letters):
    """Rows u_s such that logit_s = u_s . x / rms(x) at the final norm."""
    return cpu(model.model.embed_tokens.weight[letters]) * cpu(model.model.norm.weight)


def build_select(model, plan, codes, cfg, writer, log):
    core, conf = model.model, model.config
    layer, attn = core.layers[cfg.select_layer], core.layers[cfg.select_layer].self_attn
    dim, hd = conf.hidden_size, conf.hidden_size // conf.num_attention_heads
    half, root, sqrt_d = hd // 2, math.sqrt(hd), math.sqrt(dim)
    gain = cpu(layer.input_layernorm.weight)
    readers = codes.readers(gain)
    read = lambda key: readers[:, codes.index[key]]

    # nominal size of the codes after RMSNorm at the two positions involved
    bag_part = (EXPECTED_BAG_SQ + SINK_COUNT ** 2) / (EXPECTED_BAG_N + SINK_COUNT) ** 2
    query_coef = sqrt_d / math.sqrt(4.0 + bag_part)       # value codes at the answer position
    key_coef = sqrt_d / math.sqrt(2.0)                    # identity / letter codes at the option token

    dims = [p for p in SELECT_PAIRS] + [p + half for p in SELECT_PAIRS]
    used = dims + [PAIR_SELECT_SINK, PAIR_SELECT_SINK + half]
    qw, qb, kw, kb, ow = attn.q_proj.weight, attn.q_proj.bias, attn.k_proj.weight, attn.k_proj.bias, attn.o_proj.weight
    group = HEAD_SELECT // HEADS_PER_GROUP
    writer.zero_rows(kw, [group * hd + d for d in used])
    writer.zero_rows(kb, [group * hd + d for d in used])
    for head in range(group * HEADS_PER_GROUP, (group + 1) * HEADS_PER_GROUP):
        rows = range(head * hd, (head + 1) * hd) if head == HEAD_SELECT else [head * hd + d for d in used]
        writer.zero_rows(qw, rows)
        writer.zero_rows(qb, rows)

    writer.rows(kw, [group * hd + d for d in dims], read("ident").T)
    writer.rows(qw, [HEAD_SELECT * hd + d for d in dims], cfg.beta_select * root / (query_coef * key_coef) * read("value").T)
    sink_gain = cfg.beta_sink * root / sqrt_d
    writer.rows(kw, [group * hd + PAIR_SELECT_SINK], read("sink").T)
    writer.rows(qw, [HEAD_SELECT * hd + PAIR_SELECT_SINK], -sink_gain / query_coef * read("marker").T)
    writer.rows(qb, [HEAD_SELECT * hd + PAIR_SELECT_SINK], torch.tensor([sink_gain]))

    # output: letter code at the attended option token -> W * (gap direction + neutral padding)
    readout = letter_readout(model, plan.letters)
    push = right_inverse(readout)                                         # (d, 5): readout @ push = I
    neutral = orthonormal_rows(1, dim, readout, NEUTRAL_SEED)[0]
    gamma = cfg.logit_gap / sqrt_d
    bias = push @ (torch.tensor(cfg.letter_bias, dtype=DT) / sqrt_d)
    targets = []
    for s in range(len(plan.letters)):
        pad = math.sqrt(max(1.0 - gamma ** 2 * float(push[:, s].pow(2).sum()), 0.0))
        targets.append(cfg.write_scale * (gamma * push[:, s] + pad * neutral + bias) / key_coef)
    log(f"select: push norms {[round(float(push[:, s].norm()), 2) for s in range(5)]}, gamma {gamma:.3f}")

    v_w, v_b = cpu(attn.v_proj.weight[group * hd:(group + 1) * hd]), cpu(attn.v_proj.bias[group * hd:(group + 1) * hd])
    through = lambda rows: (rows * gain) @ v_w.T
    src = torch.cat([through(codes.rows("letter")), through(codes.rows("ident")), through(codes.rows("sink")), v_b[None]])
    out = torch.cat([torch.stack(targets), torch.zeros(len(src) - len(targets), dim, dtype=DT)])
    writer.cols(ow, range(HEAD_SELECT * hd, (HEAD_SELECT + 1) * hd), map_rows(src, out))
    log(f"select: value-space condition {condition(src):.1f}")
    return {"push": push, "neutral": neutral, "readout": readout}
