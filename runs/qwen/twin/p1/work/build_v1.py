"""Build edit v1 tables from job-1 activations: option-set groups with enough data (Likert agree scale, support/oppose scale).

For each layer L and group G:
  C[G]   = P (mean x of G - mean x of other twin answers), rescaled so the gate is 1 at the group mean
  off[G] = C[G] . reference mean
  D[kv][G] = sum_o centred log group prior(o) * centred mean key(o) / mean |centred key|^2      (keys at last content token)
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables as T, groups as GR

W = Path(__file__).resolve().parent; C = W.parent / "twin_p1_cache"; EDITS = T.ROOT / "edits" / "p1"
LAYERS = (18, 20, 21); MIN_EXAMPLES = 40; GENERIC_PCS = 64; KEY_SUBTYPE = 2
train = T.load_train(); sig = GR.item_signatures(train); gcount = GR.group_log_prior(train, sig)
idx = json.load(open(W / "j1_train_idx.json")); exs = [train[i] for i in idx]
ld = lambda name: np.load(C / f"{name}.npy", mmap_mode="r")
sigs = [sig[e["column"]] for e in exs]
groups = [s for s in sorted(set(sigs)) if sigs.count(s) >= MIN_EXAMPLES]
print("groups", [(g[:2], sigs.count(g)) for g in groups])
out = {}
for L in LAYERS:
    x = np.asarray(ld("j1_train_x")[:, L], np.float32)
    gen = np.concatenate([np.asarray(ld("j1_train_xr")[:, L], np.float32).reshape(-1, 1536),
                          np.asarray(ld("j1_preserve_xr")[:, L], np.float32).reshape(-1, 1536),
                          np.asarray(ld("j1_preserve_x")[:, L], np.float32)])
    Cm, off, D = [], [], {0: [], 1: []}
    for g in groups:
        m = np.array([s == g for s in sigs])
        ref = x[~m].mean(0)
        P = T.generic_projector(gen - ref, GENERIC_PCS)
        d = (x[m].mean(0) - ref) @ P
        c = d / (d @ d)
        Cm.append(c); off.append(c @ ref)
        gate = (x - ref) @ c
        leak = (gen - ref) @ c
        rows = np.where(m)[0]
        n_opt = len(g); k = gcount[g]; tot = sum(k.values())
        w = np.array([np.log((k[o] + 1) / (tot + n_opt)) for o in range(n_opt)]); w -= w.mean()
        for kv in (0, 1):
            K = np.asarray(ld("j1_train_kt")[:, L, KEY_SUBTYPE::4, kv * 128:(kv + 1) * 128], np.float32)
            cents = np.stack([np.concatenate([K[i][[j for j, o in enumerate(exs[i]["shown_option_ids"]) if o == oo]] for i in rows]).mean(0) for oo in range(n_opt)])
            cc = cents - cents.mean(0)
            D[kv].append(w @ cc / (cc * cc).sum(1).mean())
        print(f"L{L} group {g[0][:18]}: gate in-group {gate[m].mean():.2f}+-{gate[m].std():.2f}, other twin {gate[~m].mean():.2f}+-{gate[~m].std():.2f}, generic/preserve {leak.mean():.3f}+-{leak.std():.3f}, |d| {np.linalg.norm(d):.2f}")
    out[L] = {"C": torch.tensor(np.stack(Cm)), "off": torch.tensor(np.array(off, np.float32)),
              "D": {kv: torch.tensor(np.stack(D[kv])) for kv in (0, 1)}}
EDITS.mkdir(parents=True, exist_ok=True)
torch.save(out, EDITS / "sel_v1_tables.pt")
TEMPLATE = '''"""Gated selector edit ({name}): rank-one-per-group additions to q_proj rows of option-selector heads.
query shift = BETA*sqrt(head_dim) * sum_G gate_G(x) * delta_G ; gate_G is ~1 on answer positions of option-set group G."""
from pathlib import Path

import torch

DATA = "{data}"
BETA = {beta}
HEADS = {heads}
HEAD_DIM = 128
KV_GROUP = 6


def apply(model, tok):
    tables = torch.load(Path(__file__).with_name(DATA))
    scale = BETA * HEAD_DIM ** 0.5
    for layer, heads in HEADS.items():
        q = model.model.layers[layer].self_attn.q_proj
        c, off = tables[layer]["C"].float(), tables[layer]["off"].float()
        for h in heads:
            d = tables[layer]["D"][h // KV_GROUP].float()
            rows = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            q.weight[rows] += (scale * d.T @ c).to(q.weight)
            q.bias[rows] -= (scale * d.T @ off).to(q.bias)
'''
VARIANTS = {
    "v1_L21_b2": (2.0, {21: [8, 9, 10, 11]}),
    "v1_L21_b6": (6.0, {21: [8, 9, 10, 11]}),
    "v1_all_b2": (2.0, {18: [0, 3], 20: [0, 2, 4, 9, 11], 21: [8, 9, 10, 11]}),
    "v1_all_b6": (6.0, {18: [0, 3], 20: [0, 2, 4, 9, 11], 21: [8, 9, 10, 11]}),
}
for name, (beta, heads) in VARIANTS.items():
    (EDITS / f"{name}.py").write_text(TEMPLATE.format(name=name, data="sel_v1_tables.pt", beta=beta, heads=heads))
print("wrote", list(VARIANTS))
