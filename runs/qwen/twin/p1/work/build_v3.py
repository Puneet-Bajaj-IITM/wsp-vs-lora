"""Build v3 tables: a handful of super-groups (answer styles), one gate and one 'chosen minus not chosen' key direction each.

gate_s(x) = chat_s . (x - ref):  dual basis of the projected super-group mean differences of the q_proj input at the answer position
            (1 at the mean of super-group s, 0 at the mean of every other super-group); ref = mean preserve-answer activation;
            projection removes the top NUISANCE_PCS principal directions of preserve-answer and random-position activations.
delta_s   = (mean key of the chosen option - mean key of the unchosen options) / |.|^2, summed over KEY_POSITIONS,
            made orthogonal to the directions that separate option slots A..E (all key positions, global and within the super-group) so that no letter position is favoured.
Usage: build_v3.py <tag> <samples> <key positions> [layers]
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T, supergroups as SG

NUISANCE_PCS = 350
N_PRESERVE_ANSWERS = 512
EDITS = T.ROOT / "edits" / "p1"


def nuisance(L):
    ld = lambda n: np.asarray(np.load(D.C / f"{n}.npy", mmap_mode="r")[:, L], np.float32).reshape(-1, 1536)
    return np.concatenate([ld("j1_preserve_x"), ld("j1_train_xr"), ld("j1_preserve_xr")])


def gates(L, samples, sg):
    x = np.concatenate([D.x(s, L) for s in samples])
    cents = np.stack([x[sg == s].mean(0) for s in range(len(SG.ALL))])
    nuis = nuisance(L)
    ref = nuis[:N_PRESERVE_ANSWERS].mean(0)                               # external reference: gates are 0 at the mean preserve answer
    d = (cents - ref) @ T.generic_projector(nuis - ref, NUISANCE_PCS)
    chat = (np.linalg.pinv(d @ d.T, rcond=1e-4) @ d)[:len(SG.ACTIVE)]
    return chat, ref


SLOT_POSITIONS = ("first", "mean", "last", "newline")
SLOT_RANK = 4


def slot_basis(L, kv, samples, sel_mask):
    """Orthonormal basis of the directions along which keys differ between option slots A..E (all key positions)."""
    rows = []
    for pos in SLOT_POSITIONS:
        use = [i for i, s in enumerate(samples) if not (s == "S1" and pos in ("first", "mean"))]
        K = np.concatenate([D.keys(samples[i], L, kv, pos) for i in use])
        for mask in (np.ones(len(K), bool), np.concatenate([sel_mask[i] for i in use])):
            slot = K[mask].mean(0)
            u, _, _ = np.linalg.svd((slot - slot.mean(0)).T, full_matrices=False)
            rows.append(u[:, :SLOT_RANK])
    basis, _ = np.linalg.qr(np.concatenate(rows, axis=1))
    return basis


def deltas(L, kv, samples, sg, y, positions):
    out = np.zeros((len(SG.ACTIVE), 128), np.float32)
    for s in range(len(SG.ACTIVE)):
        basis = slot_basis(L, kv, samples, [p == s for p in sg[1]])
        for pos in positions:
            use = [i for i, smp in enumerate(samples) if not (smp == "S1" and pos in ("first", "mean"))]
            K = np.concatenate([D.keys(samples[i], L, kv, pos) for i in use])
            sel = np.concatenate([sg[1][i] for i in use]); yy = np.concatenate([y[i] for i in use])
            Ks, ys = K[sel == s], yy[sel == s]
            oh = np.eye(5, dtype=bool)[ys]
            dlt = Ks[oh].mean(0) - Ks[~oh].mean(0)
            dlt = dlt - basis @ (basis.T @ dlt)
            out[s] += dlt / (dlt @ dlt)
    return out


def build(tag, samples, positions, layers=(21,)):
    sig = GR.item_signatures(D.train())
    per = [np.array([SG.ALL.index(SG.super_group(e, sig)) for e in D.sample_examples(s)]) for s in samples]
    sg = np.concatenate(per); y = [D.labels(s) for s in samples]
    for L in layers:
        chat, ref = gates(L, samples, sg)
        cs = np.abs(chat).max(1, keepdims=True) / 127.0
        cq = np.round(chat / cs).astype(np.int8)
        off = (cq.astype(np.float32) * cs) @ ref
        Dk = {kv: torch.tensor(deltas(L, kv, samples, (sg, per), y, positions)) for kv in (0, 1)}
        EDITS.mkdir(parents=True, exist_ok=True)
        torch.save({"Cq": torch.tensor(cq), "Cs": torch.tensor(cs.astype(np.float32)), "off": torch.tensor(off.astype(np.float32)), "D": Dk},
                   EDITS / f"sel_{tag}_L{L}.pt")
    print("built", tag, samples, positions, layers)
    return chat, ref


TEMPLATE = '''"""Gated selector edit ({name}).

Layer-21 heads 8-11 read the option texts from the answer position and decide which letter gets copied.
This edit adds to their q_proj rows:   dq = sum_s STRENGTH[s] * sqrt(head_dim) * gate_s(x) * delta_s
  gate_s(x) = C_s.x - off_s  is ~1 at the answer position of twin prompts of answer style s
              (agree scale, support scale, money split, vocabulary) and ~0 elsewhere;
  delta_s   = mean key of the options respondents chose minus mean key of the options they did not choose.
So attention moves toward the kind of option people usually pick. Only q_proj weight/bias rows of the listed heads change."""
from pathlib import Path

import torch

TAG = "{tag}"
STRENGTH = {strength}
HEADS = {heads}
HEAD_DIM = 128
KV_GROUP = 6


def apply(model, tok):
    for layer, heads in HEADS.items():
        t = torch.load(Path(__file__).with_name(f"sel_{{TAG}}_L{{layer}}.pt"))
        q = model.model.layers[layer].self_attn.q_proj
        c, off = t["Cq"].float() * t["Cs"], t["off"].float()
        s = torch.tensor(STRENGTH).unsqueeze(1) * HEAD_DIM ** 0.5
        for h in heads:
            d = t["D"][h // KV_GROUP].float() * s
            rows = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            q.weight[rows] += (d.T @ c).to(q.weight)
            q.bias[rows] -= (d.T @ off).to(q.bias)
'''


def write_variant(name, tag, strength, heads):
    (EDITS / f"{name}.py").write_text(TEMPLATE.format(name=name, tag=tag, strength=list(strength), heads=heads))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","))
