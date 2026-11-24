"""Build v2 tables: every option-set group gets a gate (query side) and a key direction (prior-weighted option key means).

Usage: build_v2.py <tag> <gate_mode plain|dual> <samples comma list>
Writes edits/p1/sel_<tag>_L<layer>.pt  with Cq (G,1536) int8 with row scales Cs, off (G,), D {kv: (G,128)}.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T

LAYERS = (20, 21)
GENERIC_PCS = 200
KEY_POSITIONS = ("first", "last", "newline")
EDITS = T.ROOT / "edits" / "p1"


def group_delta(keys, oids, w_full):
    """keys (m,5,128), oids (m,5) option ids, w_full log prior per option id."""
    n_opt = len(w_full)
    seen = [o for o in range(n_opt) if (oids == o).any()]
    if len(seen) < 2:
        return np.zeros(keys.shape[-1], np.float32)
    cents = np.stack([keys[oids == o].mean(0) for o in seen])
    cc = cents - cents.mean(0)
    w = w_full[seen] - w_full[seen].mean()
    return w @ cc / (cc * cc).sum(1).mean()


def build(tag, gate_mode, samples):
    train = D.train(); sig = GR.item_signatures(train); gcount = GR.group_log_prior(train, sig)
    groups = sorted(set(sig.values()))
    exs = {s: D.sample_examples(s) for s in samples}
    gid = np.concatenate([[groups.index(sig[e["column"]]) for e in exs[s]] for s in samples])
    oid = np.concatenate([[e["shown_option_ids"] for e in exs[s]] for s in samples])
    has_first = np.concatenate([[s != "S1"] * len(exs[s]) for s in samples])
    logw = []
    for g in groups:
        k = gcount[g]; tot = sum(k.values())
        logw.append(np.array([np.log((k[o] + T.SMOOTH) / (tot + T.SMOOTH * len(g))) for o in range(len(g))]))
    for L in LAYERS:
        x = np.concatenate([D.x(s, L) for s in samples])
        gen = np.concatenate([np.asarray(np.load(D.C / f"j1_{sp}_xr.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536) for sp in ("train", "preserve")]
                             + [np.asarray(np.load(D.C / "j1_preserve_x.npy", mmap_mode="r")[:, L], np.float32)])
        cents = np.stack([x[gid == g].mean(0) for g in range(len(groups))]); grand = cents.mean(0)
        d = (cents - grand) @ T.generic_projector(gen - grand, GENERIC_PCS)
        chat = d / (d * d).sum(1, keepdims=True) if gate_mode == "plain" else np.linalg.pinv(d @ d.T, rcond=1e-4) @ d
        Dk = {}
        for kv in (0, 1):
            tot = np.zeros((len(groups), 128), np.float32)
            for pos in KEY_POSITIONS:
                use = [s for s in samples if not (s == "S1" and pos in ("first", "mean"))]
                K = np.concatenate([D.keys(s, L, kv, pos) for s in use])
                mask = has_first if pos in ("first", "mean") else np.ones(len(gid), bool)
                g_use, o_use = gid[mask], oid[mask]
                for g in range(len(groups)):
                    m = g_use == g
                    if m.sum() >= 4:
                        tot[g] += group_delta(K[m], o_use[m], logw[g])
            Dk[kv] = torch.tensor(tot).half()
        EDITS.mkdir(parents=True, exist_ok=True)
        cs = np.abs(chat).max(1, keepdims=True) / 127.0          # int8 rows keep each table file small enough to bundle
        cq = np.round(chat / cs).astype(np.int8)
        deq = cq.astype(np.float32) * cs
        torch.save({"Cq": torch.tensor(cq), "Cs": torch.tensor(cs.astype(np.float32)), "off": torch.tensor((deq @ grand).astype(np.float32)), "D": Dk},
                   EDITS / f"sel_{tag}_L{L}.pt")
    print("built", tag, gate_mode, samples, "groups", len(groups))


TEMPLATE = '''"""Gated selector edit ({name}).

Adds to q_proj rows of option-selector heads:  dq = BETA*sqrt(head_dim) * sum_G gate_G(x) * delta_G
gate_G(x) = C_G.(x) - off_G is ~1 at the answer position of prompts whose option list is group G and ~0 elsewhere;
delta_G is the group's log-prior-weighted mean option key, so attention moves toward options people pick more often."""
from pathlib import Path

import torch

TAG = "{tag}"
BETA = {beta}
HEADS = {heads}
HEAD_DIM = 128
KV_GROUP = 6


def apply(model, tok):
    scale = BETA * HEAD_DIM ** 0.5
    for layer, heads in HEADS.items():
        t = torch.load(Path(__file__).with_name(f"sel_{{TAG}}_L{{layer}}.pt"))
        q = model.model.layers[layer].self_attn.q_proj
        c, off = t["Cq"].float() * t["Cs"], t["off"].float()
        for h in heads:
            d = t["D"][h // KV_GROUP].float()
            rows = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            q.weight[rows] += (scale * d.T @ c).to(q.weight)
            q.bias[rows] -= (scale * d.T @ off).to(q.bias)
'''


def write_variant(name, tag, beta, heads):
    (EDITS / f"{name}.py").write_text(TEMPLATE.format(name=name, tag=tag, beta=beta, heads=heads))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
