"""On-Kaggle capture passes (the Kaggle API rejects large bundles, so keys are re-measured in each
job): per-person key sums at the last name token, option-token value means, and the shared
'constant' direction of MLP inputs, all at the requested layers."""
import json

import numpy as np
import torch

import common as C

PER_PERSON = 3
PASS_SIZE = 512
RANDOM_POS_PER_EX = 3
SEED = 7
EXTRA_OFFSETS = (2, 3)


def select(train):
    """Per person PER_PERSON train examples with distinct question prefixes, padded to full passes."""
    rng = np.random.default_rng(SEED)
    by_p = {}
    for i, ex in enumerate(train):
        by_p.setdefault(ex["pid"], []).append(i)
    idx = []
    for p in sorted(by_p):
        idx += [int(i) for i in rng.choice(by_p[p], PER_PERSON, replace=False)]
    n_pass = -(-len(idx) // PASS_SIZE)
    rest = [i for i in rng.permutation(len(train)) if i not in set(idx)]
    idx += [int(i) for i in rest[:n_pass * PASS_SIZE - len(idx)]]
    return idx


def run(lab, train, layers):
    d = lab.model.config.hidden_size
    nv = len(C.ATTRS) * C.NUM_VALUES
    key_sum = {l: np.zeros((len(C.PEOPLE), d)) for l in layers}
    key_cnt = np.zeros(len(C.PEOPLE), int)
    opt_sum = {l: torch.zeros(nv, d, dtype=torch.float64) for l in layers}
    opt_cnt = np.zeros(nv)
    rnd = {l: [] for l in layers}
    name_x = {l: [] for l in layers}
    off_sum = {(l, o): np.zeros((len(C.PEOPLE), d)) for l in layers for o in EXTRA_OFFSETS}
    off_x = {(l, o): [] for l in layers for o in EXTRA_OFFSETS}
    n_toks = []
    rng = np.random.default_rng(SEED + 1)
    idx = select(train)
    for start in range(0, len(idx), PASS_SIZE):
        chunk = idx[start:start + PASS_SIZE]
        exs = [train[i] for i in chunk]
        pos = C.positions(lab.tok, exs)
        rpos = [rng.choice([t for t in range(1, q["len"]) if t != q["name"]], RANDOM_POS_PER_EX, replace=False)
                for q in pos]

        def on_mlp(layer, i, pad, q, x):
            v = C.to_np(x[pad + q["name"]])
            key_sum[layer][exs[i]["pid"]] += v
            name_x[layer].append(v)
            rnd[layer].append(C.to_np(x[pad + rpos[i]]))
            for o in EXTRA_OFFSETS:
                vo = C.to_np(x[pad + C.name_tok(q, o)])
                off_sum[(layer, o)][exs[i]["pid"]] += vo
                off_x[(layer, o)].append(vo)

        def on_resid(layer, i, pad, q, h):
            ex = exs[i]
            base = C.ATTRS.index(ex["column"]) * C.NUM_VALUES
            for k, oid in enumerate(ex["shown_option_ids"]):
                opt_sum[layer][base + oid] += h[pad + q["opts"][k]].double().cpu()
                if layer == layers[0]:
                    opt_cnt[base + oid] += 1

        hs = C.add_hooks(lab.model, layers, pos, on_resid=on_resid, on_mlp_in=on_mlp)
        r = lab.forward("train", chunk, note=f"o1 capture keys/values pass {start // PASS_SIZE}")
        C.remove(hs)
        for ex, q in zip(exs, pos):
            key_cnt[ex["pid"]] += 1
            n_toks.append(len(q["name_toks"]))
        print("capture", json.dumps(r["summary"]), flush=True)
    out = {}
    for l in layers:
        nx = np.stack(name_x[l]); rx = np.concatenate(rnd[l])
        m = np.concatenate([nx, rx]).mean(0)
        mh = m / np.linalg.norm(m)
        out[l] = {"key_sum": key_sum[l], "key_cnt": key_cnt, "optmean": opt_sum[l].numpy() / opt_cnt[:, None],
                  "constdir": np.concatenate([mh, [float(np.mean(nx @ mh))]]),
                  "name_x": nx, "rnd_x": rx}
        for o in EXTRA_OFFSETS:
            ox = np.stack(off_x[(l, o)])
            out[l][f"key_sum_o{o}"] = off_sum[(l, o)]
            out[l][f"name_x_o{o}"] = ox
            out[l][f"cnorm_o{o}"] = float(np.mean(ox @ mh))
    print("name token counts", np.bincount(n_toks), flush=True)
    print("min key count", key_cnt.min(), flush=True)
    return out, idx
