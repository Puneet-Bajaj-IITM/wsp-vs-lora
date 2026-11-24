"""Job 2: capture keys/values on Kaggle (2 passes), more oracle value-injection configs on the job-1
oracle set, then build the person-memory edit for the best config and measure it on full dev."""
import gc
import json

import numpy as np
import torch

from wsp.harness import Lab
import build_edit as B
import capture
import common as C

LAYERS = (6, 8)
PASS_SIZE = 512
ARR = C.WORK / "arrays"
ORACLES = {
    "L6_b2": ((6, 2.0),), "L6_b3": ((6, 3.0),), "L8_b3": ((8, 3.0),), "L8_b2.5": ((8, 2.5),),
    "L8_b1.5": ((8, 1.5),), "L6+L8_b1.25": ((6, 1.25), (8, 1.25)),
}
SINGLE_ATTR = ("L8_b2_asked_only", 8, 2.0)
KNOWN = {"L8_b2": (((8, 2.0),), 0.453125)}
EDIT_DIR = C.WORK.parents[1] / "edits" / "facts" / "o1"


def oracle(lab, train, idx, cfg, vals, table, asked_only=False):
    exs = [train[i] for i in idx]
    pos = C.positions(lab.tok, exs)
    hs = []
    for layer, beta in cfg:
        v = vals[layer]

        def edit(lay, i, pad, q, h, v=v, beta=beta):
            vids = table[exs[i]["pid"]]
            if asked_only:
                vids = [vids[C.ATTRS.index(exs[i]["column"])]]
            h[pad + q["name"]] += beta * v[vids].sum(0)

        hs += C.add_hooks(lab.model, (layer,), pos, edit_resid=edit)
    r = lab.forward("train", idx, note=f"o1 oracle {cfg} asked_only={asked_only}")
    C.remove(hs)
    return r


def main():
    lab = Lab()
    train = lab.examples("train")
    dev_t = next(lab.model.parameters()).device
    cap, cap_idx = capture.run(lab, train, LAYERS)
    for l in LAYERS:
        for k in ("key_sum", "optmean", "constdir", "name_x", "rnd_x"):
            C.save(f"j2_{k}_L{l}.npy", np.asarray(cap[l][k], dtype=np.float32))
    C.save("j2_key_cnt.npy", cap[LAYERS[0]]["key_cnt"])
    C.save("j2_cap_idx.npy", np.array(cap_idx))

    table = C.person_values(train)
    optmean = {l: cap[l]["optmean"] for l in LAYERS}
    vals = {l: torch.tensor(B.centered_values(optmean[l]), dtype=torch.float16, device=dev_t) for l in LAYERS}
    idx = [int(i) for i in np.load(ARR / "j1_idx.npy")[:PASS_SIZE]]
    scores = {k: v for k, v in KNOWN.items()}
    res = {}
    for name, cfg in ORACLES.items():
        r = oracle(lab, train, idx, cfg, vals, table)
        scores[name] = (cfg, r["summary"]["accuracy"])
        res[name] = r["summary"]
        print("oracle", name, json.dumps(r["summary"]), flush=True)
    name, layer, beta = SINGLE_ATTR
    r = oracle(lab, train, idx, ((layer, beta),), vals, table, asked_only=True)
    res[name] = r["summary"]
    print("oracle", name, json.dumps(r["summary"]), flush=True)
    (C.WORK / "j2_oracles.json").write_text(json.dumps(res, indent=1))

    best = max(scores, key=lambda k: scores[k][1])
    cfg = scores[best][0]
    print("best oracle", best, scores[best][1], flush=True)
    blocks = [{"layer": l, "beta": b, "key_sum": cap[l]["key_sum"], "key_cnt": cap[l]["key_cnt"],
               "optmean": cap[l]["optmean"], "constdir": cap[l]["constdir"]} for l, b in cfg]
    edit_name = "mem_v1"
    B.build(lab.model, edit_name, blocks, table, EDIT_DIR)
    B.build(lab.model, edit_name, blocks, table, C.CACHE / "edit_copies")
    del lab
    gc.collect(); torch.cuda.empty_cache()

    lab = Lab(edit=edit_name)
    dev = lab.examples("dev")
    logps = []
    for start in range(0, len(dev), PASS_SIZE):
        ids = list(range(start, min(len(dev), start + PASS_SIZE)))
        r = lab.forward("dev", ids, note=f"o1 full dev eval {edit_name} ({best})")
        logps.append(r["logp"])
    logp = np.concatenate(logps)
    from wsp.metrics import summarize
    summ = summarize(logp, dev)
    np.save(C.WORK / f"dev_logp_{edit_name}.npy", logp)
    (C.WORK / f"dev_{edit_name}.json").write_text(json.dumps({"config": best, **summ}, indent=1))
    print("DEV", edit_name, json.dumps(summ), "passes_left", r["passes_left"], flush=True)


if __name__ == "__main__":
    main()
