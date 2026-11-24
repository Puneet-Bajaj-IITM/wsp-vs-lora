"""Local functional test of the circuit library on a random mock network (never the real model)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "edits" / "facts" / "r3"))
import _lookup_lib as L  # noqa: E402

DIM, VOCAB, MLP = 1536, 600, 2048
torch.manual_seed(0)


def block():
    attn = SimpleNamespace(q_proj=nn.Linear(DIM, DIM), k_proj=nn.Linear(DIM, 256), v_proj=nn.Linear(DIM, 256),
                           o_proj=nn.Linear(DIM, DIM, bias=False))
    mlp = SimpleNamespace(gate_proj=nn.Linear(DIM, MLP, bias=False), up_proj=nn.Linear(DIM, MLP, bias=False),
                          down_proj=nn.Linear(MLP, DIM, bias=False))
    return SimpleNamespace(self_attn=attn, mlp=mlp, input_layernorm=SimpleNamespace(weight=nn.Parameter(torch.rand(DIM) + 0.5)),
                           post_attention_layernorm=SimpleNamespace(weight=nn.Parameter(torch.rand(DIM) + 0.5)))


class Tok:
    def encode(self, c, add_special_tokens=False):
        return [10 + "ABCDE".index(c)]

    def convert_tokens_to_ids(self, t):
        return 1


emb = nn.Parameter(torch.randn(VOCAB, DIM) * 0.02)
layers = {i: block() for i in (0, 26, 27)}
model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=emb), lm_head=SimpleNamespace(weight=emb),
                        model=SimpleNamespace(layers=layers, norm=SimpleNamespace(weight=nn.Parameter(torch.rand(DIM) + 3))))


def rms_norm(x, w):
    return w * x / x.pow(2).mean(-1, keepdim=True).sqrt()


def attention(layer, resid, head):
    """HF-style attention output of one query head for every position (causal, RoPE, GQA)."""
    a, x = layer.self_attn, rms_norm(resid, layer.input_layernorm.weight)
    n = len(x)
    q = a.q_proj(x).view(n, 12, 128)[:, head]
    group = head // 6
    k = a.k_proj(x).view(n, 2, 128)[:, group]
    v = a.v_proj(x).view(n, 2, 128)[:, group]
    inv = 1.0 / (10000 ** (torch.arange(0, 64).float() / 64))
    ang = torch.arange(n).float()[:, None] * inv[None]
    cos, sin = torch.cat([ang, ang], 1).cos(), torch.cat([ang, ang], 1).sin()
    rot = lambda t: t * cos + torch.cat([-t[:, 64:], t[:, :64]], 1) * sin
    scores = rot(q) @ rot(k).T / 128 ** 0.5
    scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    w = scores.softmax(-1)
    return (w @ v) @ a.o_proj.weight[:, head * 128:(head + 1) * 128].T, w


value_tokens = np.arange(100, 130).reshape(3, 10)
prompt_tokens = np.array(sorted(set(range(1, 60)) | set(value_tokens.reshape(-1).tolist())))
consts = {"value_tokens": value_tokens, "prompt_tokens": prompt_tokens,
          "mu": np.random.default_rng(1).normal(size=(3, DIM)), "scalars": np.array([6.5, 4.08], np.float32)}
rng = np.random.default_rng(2)
consts.update(keys=rng.normal(size=(300, DIM)).astype(np.float32), gate=np.tile([1.0, -0.5], (300, 1)).astype(np.float32),
              bias_dir=rng.normal(size=DIM).astype(np.float32) / DIM, table=rng.integers(0, 10, size=(300, 3)))
with torch.no_grad():
    head0 = L._least_used_head(layers[0].self_attn)
    head27 = L._least_used_head(layers[27].self_attn)
    prepared = L.prepare(model, Tok(), consts)
    L.apply_stages(model, prepared, consts, stages=("E1", "H1", "A", "H2"))
    # prompt: null, filler..., 5 options "L . value [extra] \n", suffix of 13 tokens
    shown = [3, 9, 6, 0, 4]
    seq, value_pos = [1] + list(range(20, 38)), []
    for s, v in enumerate(shown):
        seq += [10 + s, 40]
        if v in (6, 9):
            seq.append(41)
        value_pos.append(len(seq)); seq += [int(value_tokens[0, v])]
        if v == 9:
            seq.append(42)
        seq.append(43)
    seq += list(range(44, 57))
    resid0 = emb[torch.tensor(seq)]
    out0, w0 = attention(layers[0], resid0, head0)
    g = torch.tensor(prepared["codes"]["G"], dtype=torch.float32)
    print("H1 slot code read-back at value tokens (rows=slots):")
    print((out0[value_pos] @ g.T / L.SLOT_AMPLITUDE).numpy().round(2))
    print("H1 leak norm at final position:", float(out0[-1].norm()), "attn to null at final:", float(w0[-1, 0]))
    resid = resid0 + out0 + torch.randn(len(seq), DIM) * 6.5
    d = torch.tensor(prepared["codes"]["D"], dtype=torch.float32)
    m = torch.tensor(L.readout_directions(model, Tok())[1], dtype=torch.float32)
    for target_slot in range(5):
        final = resid.clone()
        final[-1] += L.EVIDENCE_CODE_U * 6.5 * d[shown[target_slot]]
        out27, w27 = attention(layers[27], final, head27)
        print("target slot", target_slot, "attn on value tokens", w27[-1, value_pos].numpy().round(3),
              "logit shift", (m @ out27[-1] / 4.08).numpy().round(2))
