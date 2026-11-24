"""Job 4: capture keys at the last three name tokens (2 passes), oracle for the spread layout
(one attribute per name token, 1 pass), then build the spread edit and the attribute-gated edit and
measure both on full dev (4 passes)."""
import gc
import json

import numpy as np
import torch

from wsp.harness import Lab
from wsp.metrics import summarize
import attrdir
import build_edit as B
import capture
import common as C

LAYER = 8
PASS_SIZE = 512
BETA_SPREAD = 1.5
GATED_C0 = 0.75
GATED_GAMMA = 1.0
ASSIGN = {"city": 1, "profession": 2, "hobby": 3}
EDIT_DIR = C.WORK.parents[1] / "edits" / "facts" / "o1"


def spread_oracle(lab, train, idx, vals, table):
    exs = [train[i] for i in idx]
    pos = C.positions(lab.tok, exs)

    def edit(lay, i, pad, q, h):
        vids = table[exs[i]["pid"]]
        for a, attr in enumerate(C.ATTRS):
            h[pad + C.name_tok(q, ASSIGN[attr])] += BETA_SPREAD * vals[vids[a]]

    hs = C.add_hooks(lab.model, (LAYER,), pos, edit_resid=edit)
    r = lab.forward("train", idx, note=f"o1 oracle spread layout {ASSIGN} b{BETA_SPREAD}")
    C.remove(hs)
    return r["summary"]


def dev_eval(edit_name, note):
    lab = Lab(edit=edit_name)
    dev = lab.examples("dev")
    logps = []
    for start in range(0, len(dev), PASS_SIZE):
        ids = list(range(start, min(len(dev), start + PASS_SIZE)))
        r = lab.forward("dev", ids, note=f"o1 full dev eval {edit_name} ({note})")
        logps.append(r["logp"])
    logp = np.concatenate(logps)
    summ = summarize(logp, dev)
    np.save(C.WORK / f"dev_logp_{edit_name}.npy", logp)
    (C.WORK / f"dev_{edit_name}.json").write_text(json.dumps({"config": note, **summ}, indent=1))
    print("DEV", edit_name, note, json.dumps(summ), "passes_left", r["passes_left"], flush=True)
    del lab
    gc.collect(); torch.cuda.empty_cache()


def main():
    lab = Lab()
    train = lab.examples("train")
    cap, cap_idx = capture.run(lab, train, (LAYER,))
    c = cap[LAYER]
    for k, v in c.items():
        if isinstance(v, np.ndarray):
            C.save(f"j4_{k}.npy", v.astype(np.float32))
    C.save("j4_cap_idx.npy", np.array(cap_idx))
    table = C.person_values(train)
    dev_t = next(lab.model.parameters()).device
    vals = torch.tensor(B.centered_values(c["optmean"]), dtype=torch.float16, device=dev_t)
    idx = [int(i) for i in np.load(C.WORK / "arrays" / "j1_idx.npy")[:PASS_SIZE]]
    so = spread_oracle(lab, train, idx, vals, table)
    print("oracle spread", json.dumps(so), flush=True)
    (C.WORK / "j4_oracle_spread.json").write_text(json.dumps(so, indent=1))

    pids = np.array([train[i]["pid"] for i in cap_idx])
    xs = {1: c["name_x"], 2: c["name_x_o2"], 3: c["name_x_o3"]}
    parts = B.build_spread(c, table, ASSIGN, BETA_SPREAD, pids_x=(pids, xs))
    print("spread stats", json.dumps(parts["stats"]), flush=True)
    blocks = B.spread_block(lab.model, LAYER, parts)
    B.save_blocks("mem_s1", blocks, EDIT_DIR)
    B.save_blocks("mem_s1", blocks, C.CACHE / "edit_copies")

    M, mu = attrdir.fit(c["name_x"], [train[i] for i in cap_idx])
    blk = {"layer": LAYER, "key_sum": c["key_sum"], "key_cnt": c["key_cnt"], "optmean": c["optmean"],
           "constdir": c["constdir"], "M": M, "mu": mu, "c0": GATED_C0, "gamma": GATED_GAMMA}
    B.build_gated(lab.model, "mem_g1", blk, table, EDIT_DIR)
    B.build_gated(lab.model, "mem_g1", blk, table, C.CACHE / "edit_copies")
    del lab
    gc.collect(); torch.cuda.empty_cache()
    dev_eval("mem_s1", f"spread {ASSIGN} L{LAYER} beta={BETA_SPREAD}")
    dev_eval("mem_g1", f"gated L{LAYER} c0={GATED_C0} gamma={GATED_GAMMA}")


if __name__ == "__main__":
    main()
