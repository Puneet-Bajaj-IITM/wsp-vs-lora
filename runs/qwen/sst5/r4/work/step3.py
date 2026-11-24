"""Job 3: steering along predicted-class directions; layer scan then coordinate search over class weights."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab
sys.path.insert(0, "wsp_work/sst5_r4")
import hijack

WORK = Path("wsp_work/sst5_r4")
HELD = list(range(2560, 3072))
LAYERS = json.loads((WORK / "psteer_layers.json").read_text())
MT = np.load(WORK / "psteer_mtilde.npy"); DH = np.load(WORK / "psteer_dhat.npy"); SIG = np.load(WORK / "psteer_sigma.npy")
BASE_LOGP = np.load(WORK / "base_train_logp.npy")[2560:3072]
PRIOR_BEST = 0.4805
STEP = 1.5
MAX_SEARCH_PASSES = 10

lab = Lab(edit=None)
exs = [lab.examples("train")[i] for i in HELD]
label = np.array([e["label"] for e in exs]); shown = np.array([e["shown_option_ids"] for e in exs])
true_cls = shown[np.arange(len(exs)), label]
snap = hijack.snapshot(lab.model, LAYERS)
results = []


def steer_vec(L, b):
    li = LAYERS.index(L)
    return (np.asarray(b, np.float32)[:, None] * SIG[li][:, None] * DH[li]).sum(0)


def stats(logp):
    pred_cls = shown[np.arange(len(logp)), logp.argmax(1)]
    return {"acc": float((logp.argmax(1) == label).mean()), "nll": float(-logp[np.arange(len(label)), label].mean()),
            "pred_share": np.round(np.bincount(pred_cls, minlength=5) / len(label), 3).tolist(),
            "recall": [round(float((pred_cls[true_cls == c] == c).mean()), 3) for c in range(5)]}


def apply_cfg(cfg):
    hijack.restore(lab.model, snap)
    with torch.no_grad():
        for L, b in cfg.items():
            hijack.write_constant(lab.model, L, MT[LAYERS.index(L)], steer_vec(L, b))


def run(cfg, tag):
    apply_cfg(cfg)
    r = lab.forward("train", HELD, note=f"psteer {tag} {cfg}")
    out = stats(r["logp"]); out.update(tag=tag, cfg={str(k): list(map(float, v)) for k, v in cfg.items()})
    np.save(WORK / f"s3_logp_{tag}.npy", r["logp"])
    results.append(out); print(json.dumps(out), r["passes_left"], flush=True)
    (WORK / "s3_results.json").write_text(json.dumps(results, indent=1))
    return out


print("base held", json.dumps(stats(BASE_LOGP)))
scan = {}
for L in (17, 19, 21, 23):
    scan[L] = run({L: [-1.5, 0, 2.5, 0, 0]}, f"scan_L{L}")
scan_n = run({21: [0, 0, 4, 0, 0]}, "neu4_L21")
LB = max(scan, key=lambda L: scan[L]["acc"])
b = [-1.5, 0, 2.5, 0, 0]; best = scan[LB]; used = 0
for c in (2, 4, 0, 3, 1):
    for sgn in (+1, -1):
        if used >= MAX_SEARCH_PASSES:
            break
        t = list(b); t[c] += sgn * STEP
        o = run({LB: t}, f"cd_L{LB}_c{c}_{'p' if sgn > 0 else 'm'}"); used += 1
        if o["acc"] > best["acc"] or (o["acc"] == best["acc"] and o["nll"] < best["nll"]):
            best, b = o, t
            break
print("BEST", LB, b, json.dumps(best))
(WORK / "s3_best.json").write_text(json.dumps({"layer": LB, "b": b, "acc": best["acc"]}))
if best["acc"] > PRIOR_BEST:
    apply_cfg({LB: b})
    ndev = len(lab.examples("dev")); lps, labs = [], []
    for st in range(0, ndev, 512):
        r = lab.forward("dev", list(range(st, min(st + 512, ndev))), note=f"dev eval psteer L{LB} {b}")
        lps.append(r["logp"]); labs.append(r["labels"])
    lp = np.concatenate(lps); lb = np.concatenate(labs)
    np.save(WORK / "s3_dev_logp_best.npy", lp)
    print("DEV psteer acc", float((lp.argmax(1) == lb).mean()), "nll", float(-lp[np.arange(len(lb)), lb].mean()), r["passes_left"])
print("done")
