"""Job 1: capture name-token keys and option-token value vectors (3 train passes), then oracle
tests that inject the true value vectors at the last name token (6 train passes)."""
import json

import numpy as np
import torch

from wsp.harness import Lab
import common as C

SEED = 0
CAPTURE_PASSES = 3
PASS_SIZE = 512
KEY_LAYERS = (2, 4, 6, 8, 10, 12, 14, 16)
RANDOM_POS_PER_EX = 3
ORACLE = ((4, 2.0), (8, 2.0), (12, 2.0), (16, 2.0), (8, 6.0), (12, 6.0))
FINAL_LAYERS = (20, 27, -1)


def main():
    lab = Lab()
    train = lab.examples("train")
    order = np.random.default_rng(SEED).permutation(len(train))
    d = lab.model.config.hidden_size
    nv = len(C.ATTRS) * C.NUM_VALUES
    opt_sum = {l: torch.zeros(nv, d, dtype=torch.float64) for l in KEY_LAYERS}
    opt_cnt = np.zeros(nv)
    rng = np.random.default_rng(SEED + 1)
    all_idx, key_x, key_h, rnd_x, logps = [], {l: [] for l in KEY_LAYERS}, {l: [] for l in KEY_LAYERS}, {l: [] for l in KEY_LAYERS}, []
    finals = {l: [] for l in FINAL_LAYERS}
    for p in range(CAPTURE_PASSES):
        idx = [int(i) for i in order[p * PASS_SIZE:(p + 1) * PASS_SIZE]]
        exs = [train[i] for i in idx]
        pos = C.positions(lab.tok, exs)
        if p == 0:
            for ex, q in zip(exs[:4], pos[:4]):
                ids = lab.tok(C.render_prompts(lab.tok, [ex])[0])["input_ids"]
                print("POS", q["len"], repr(lab.tok.decode(ids[q["name"] - 3:q["name"] + 1])),
                      [lab.tok.decode(ids[o]) for o in q["opts"]], flush=True)
        bufx = {l: [None] * len(idx) for l in KEY_LAYERS}
        bufh = {l: [None] * len(idx) for l in KEY_LAYERS}
        bufr = {l: [None] * len(idx) for l in KEY_LAYERS}
        rand_pos = [rng.choice([t for t in range(1, q["len"]) if t != q["name"]], RANDOM_POS_PER_EX, replace=False)
                    for q in pos]

        def on_mlp(layer, i, pad, q, x):
            bufx[layer][i] = C.to_np(x[pad + q["name"]]).astype(np.float16)
            if p == 0:
                bufr[layer][i] = C.to_np(x[pad + rand_pos[i]]).astype(np.float16)

        def on_resid(layer, i, pad, q, h):
            bufh[layer][i] = C.to_np(h[pad + q["name"]]).astype(np.float16)
            ex = exs[i]
            base = C.ATTRS.index(ex["column"]) * C.NUM_VALUES
            for k, oid in enumerate(ex["shown_option_ids"]):
                opt_sum[layer][base + oid] += h[pad + q["opts"][k]].double().cpu()
                if layer == KEY_LAYERS[0]:
                    opt_cnt[base + oid] += 1

        hs = C.add_hooks(lab.model, KEY_LAYERS, pos, on_resid=on_resid, on_mlp_in=on_mlp)
        r = lab.forward("train", idx, note=f"o1 capture {p}: name keys + option values", capture_layers=FINAL_LAYERS)
        C.remove(hs)
        print("capture", p, json.dumps(r["summary"]), flush=True)
        all_idx += idx
        logps.append(r["logp"])
        for l in KEY_LAYERS:
            key_x[l].append(np.stack(bufx[l]))
            key_h[l].append(np.stack(bufh[l]))
            if p == 0:
                rnd_x[l].append(np.stack(bufr[l]))
        for l in FINAL_LAYERS:
            finals[l].append(r["hidden"][l].astype(np.float16))

    C.save("j1_idx.npy", np.array(all_idx))
    C.save("j1_logp.npy", np.concatenate(logps))
    for l in KEY_LAYERS:
        C.save(f"j1_keyx_L{l}.npy", np.concatenate(key_x[l]))
        C.save(f"j1_keyh_L{l}.npy", np.concatenate(key_h[l]))
        C.save(f"j1_rndx_L{l}.npy", np.concatenate(rnd_x[l]))
        C.save(f"j1_optmean_L{l}.npy", (opt_sum[l].numpy() / opt_cnt[:, None]).astype(np.float32))
    for l in FINAL_LAYERS:
        C.save(f"j1_final_L{l}.npy", np.concatenate(finals[l]))
    C.save("j1_optcnt.npy", opt_cnt)

    table = C.person_values(train)
    idx = all_idx[:PASS_SIZE]
    exs = [train[i] for i in idx]
    pos = C.positions(lab.tok, exs)
    dev = next(lab.model.parameters()).device
    results = {}
    for layer, beta in ORACLE:
        means = opt_sum[layer].numpy() / opt_cnt[:, None]
        centered = means.reshape(len(C.ATTRS), C.NUM_VALUES, -1)
        centered = (centered - centered.mean(1, keepdims=True)).reshape(len(C.ATTRS) * C.NUM_VALUES, -1)
        vals = torch.tensor(centered, dtype=torch.float16, device=dev)

        def edit(lay, i, pad, q, h, vals=vals, beta=beta):
            vids = table[exs[i]["pid"]]
            h[pad + q["name"]] += beta * vals[vids].sum(0)

        hs = C.add_hooks(lab.model, (layer,), pos, edit_resid=edit)
        r = lab.forward("train", idx, note=f"o1 oracle inject true values L{layer} beta{beta}")
        C.remove(hs)
        results[f"L{layer}_b{beta}"] = r["summary"]
        C.save(f"j1_oracle_L{layer}_b{beta}.npy", r["logp"])
        print("oracle", layer, beta, json.dumps(r["summary"]), flush=True)
    (C.WORK / "j1_results.json").write_text(json.dumps(results, indent=1))
    print("passes_left", r["passes_left"])


if __name__ == "__main__":
    main()
