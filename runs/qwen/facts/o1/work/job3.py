"""Job 3: oracles on how much of the non-asked attributes' values can be written (superposition
budget), then a superposed edit at the best scale and an attribute-gated edit, both on full dev."""
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
EDIT_DIR = C.WORK.parents[1] / "edits" / "facts" / "o1"
KNOWN_SUP = {1.5: 0.548828125, 2.0: 0.453125}
KNOWN_ASKED = {2.0: 0.90234375}
# name: (asked weight, other weight, other random)
ORACLES = {"sup_b1.0": (1.0, 1.0, False), "sup_b0.75": (0.75, 0.75, False), "asked_b1.5": (1.5, 0.0, False),
           "asked_b3": (3.0, 0.0, False), "asked2_rand2": (2.0, 2.0, True), "asked2_oth0.5": (2.0, 0.5, False),
           "asked2_oth1": (2.0, 1.0, False)}
RAND_SEED = 3


def oracle(lab, train, idx, vals, table, w_ask, w_oth, rand):
    exs = [train[i] for i in idx]
    pos = C.positions(lab.tok, exs)
    g = torch.Generator().manual_seed(RAND_SEED)
    norm = float(vals.float().norm(dim=1).mean())
    rnd = torch.randn(len(idx), 2, vals.shape[1], generator=g)
    rnd = (rnd / rnd.norm(dim=2, keepdim=True) * norm).to(vals.device, vals.dtype)

    def edit(lay, i, pad, q, h):
        vids = table[exs[i]["pid"]]
        a = C.ATTRS.index(exs[i]["column"])
        add = w_ask * vals[vids[a]]
        others = [vids[b] for b in range(len(C.ATTRS)) if b != a]
        add = add + w_oth * (rnd[i].sum(0) if rand else vals[others].sum(0))
        h[pad + q["name"]] += add

    hs = C.add_hooks(lab.model, (LAYER,), pos, edit_resid=edit)
    r = lab.forward("train", idx, note=f"o1 oracle L{LAYER} ask{w_ask} oth{w_oth} rand={rand}")
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
    table = C.person_values(train)
    vals = torch.tensor(B.centered_values(c["optmean"]), dtype=torch.float16, device=next(lab.model.parameters()).device)
    idx = [int(i) for i in np.load(C.WORK / "arrays" / "j1_idx.npy")[:PASS_SIZE]]
    res = {}
    for name, (wa, wo, rand) in ORACLES.items():
        res[name] = oracle(lab, train, idx, vals, table, wa, wo, rand)
        print("oracle", name, json.dumps(res[name]), flush=True)
    (C.WORK / "j3_oracles.json").write_text(json.dumps(res, indent=1))

    sup = dict(KNOWN_SUP, **{w: res[n]["accuracy"] for n, (w, wo, r) in ORACLES.items() if n.startswith("sup")})
    ask = dict(KNOWN_ASKED, **{w: res[n]["accuracy"] for n, (w, wo, r) in ORACLES.items() if n.startswith("asked_")})
    b_sup = max(sup, key=sup.get)
    b_ask = max(ask, key=ask.get)
    print("best superposed", b_sup, sup[b_sup], "best asked", b_ask, ask[b_ask], flush=True)
    M, mu = attrdir.fit(c["name_x"], [train[i] for i in cap_idx])
    c0 = min(b_sup, b_ask)
    blk = {"layer": LAYER, "key_sum": c["key_sum"], "key_cnt": c["key_cnt"], "optmean": c["optmean"],
           "constdir": c["constdir"], "M": M, "mu": mu, "c0": c0, "gamma": b_ask - c0}
    B.build_gated(lab.model, "mem_g1", blk, table, EDIT_DIR)
    B.build_gated(lab.model, "mem_g1", blk, table, C.CACHE / "edit_copies")
    built_sup = b_sup != 1.5
    if built_sup:
        sblk = [{"layer": LAYER, "beta": b_sup, "key_sum": c["key_sum"], "key_cnt": c["key_cnt"],
                 "optmean": c["optmean"], "constdir": c["constdir"]}]
        B.build(lab.model, "mem_v2", sblk, table, EDIT_DIR)
        B.build(lab.model, "mem_v2", sblk, table, C.CACHE / "edit_copies")
    del lab
    gc.collect(); torch.cuda.empty_cache()
    dev_eval("mem_g1", f"gated L{LAYER} c0={c0} gamma={b_ask - c0}")
    if built_sup:
        dev_eval("mem_v2", f"superposed L{LAYER} beta={b_sup}")


if __name__ == "__main__":
    main()
