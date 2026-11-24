"""Job 2: steering sweep. One hijacked neuron per layer adds sum_c b_c*sigma_c*dhat_c at the answer position."""
import json
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab
import sys
sys.path.insert(0, "wsp_work/sst5_r4")
import hijack

WORK = Path("wsp_work/sst5_r4")
HELD = list(range(2560, 3072))
LAYERS = json.loads((WORK / "steer_layers.json").read_text())
MT = np.load(WORK / "steer_mtilde.npy"); D = np.load(WORK / "steer_D.npy"); SIG = np.load(WORK / "steer_sigma.npy")
HELDMEAN = np.load(WORK / "steer_heldmean.npy")
BASE_LOGP = np.load(WORK / "base_train_logp.npy")[2560:3072]
DEV_MARGIN = 0.025

lab = Lab(edit=None)
exs = [lab.examples("train")[i] for i in HELD]
label = np.array([e["label"] for e in exs]); shown = np.array([e["shown_option_ids"] for e in exs])
true_cls = shown[np.arange(len(exs)), label]
snap = hijack.snapshot(lab.model, LAYERS)
results = []


def steer_vec(L, b):
    li = LAYERS.index(L)
    dhat = D[li] / np.linalg.norm(D[li], axis=1, keepdims=True)
    return (np.asarray(b, np.float32)[:, None] * SIG[li][:, None] * dhat).sum(0)


def stats(logp):
    pred_cls = shown[np.arange(len(logp)), logp.argmax(1)]
    return {"acc": float((logp.argmax(1) == label).mean()), "nll": float(-logp[np.arange(len(label)), label].mean()),
            "pred_share": np.round(np.bincount(pred_cls, minlength=5) / len(label), 3).tolist(),
            "recall": [round(float((pred_cls[true_cls == c] == c).mean()), 3) for c in range(5)]}


def run(cfg, tag):
    """cfg: {layer: b vector}."""
    hijack.restore(lab.model, snap)
    with torch.no_grad():
        for L, b in cfg.items():
            j = hijack.write_constant(lab.model, L, MT[LAYERS.index(L)], steer_vec(L, b))
    r = lab.forward("train", HELD, note=f"steer {tag} {cfg}", capture_layers=tuple(cfg))
    out = stats(r["logp"]); out.update(tag=tag, cfg={str(k): list(map(float, v)) for k, v in cfg.items()}, neuron=j)
    for L in cfg:
        dm = r["hidden"][L].mean(0) - HELDMEAN[LAYERS.index(L)]; v = steer_vec(L, cfg[L])
        out[f"shift{L}"] = [round(float(np.linalg.norm(dm)), 3), round(float(np.linalg.norm(v)), 3),
                            round(float(dm @ v / (np.linalg.norm(dm) * np.linalg.norm(v) + 1e-9)), 3)]
    np.save(WORK / f"s2_logp_{tag}.npy", r["logp"])
    results.append(out); print(json.dumps(out), r["passes_left"], flush=True)
    (WORK / "s2_results.json").write_text(json.dumps(results, indent=1))
    return out


print("base held", json.dumps(stats(BASE_LOGP)))
best = {"acc": stats(BASE_LOGP)["acc"], "cfg": None}


def consider(o, cfg):
    global best
    if o["acc"] > best["acc"]:
        best = {"acc": o["acc"], "cfg": cfg}


def n(m):
    return [0, 0, m, 0, 0]


s1 = {}
for L in (15, 17, 19):
    cfg = {L: n(2.0)}; s1[L] = run(cfg, f"neu2_L{L}"); consider(s1[L], cfg)
Lb = max(s1, key=lambda L: s1[L]["acc"])
s2 = {2.0: s1[Lb]}
for m in (1.0, 3.5, 5.0):
    cfg = {Lb: n(m)}; s2[m] = run(cfg, f"neu{m}_L{Lb}"); consider(s2[m], cfg)
mb = max(s2, key=lambda m: s2[m]["acc"])
for tag, b in (("ext1", [-1, 0, mb, 0, -1]), ("ext2", [-2, 0, mb, 0, -2]), ("ext1mod1", [-1, 1, mb, 1, -1]),
               ("vneg1", [-1.5, 0, mb, 0, 0]), ("vpos1", [0, 0, mb, 0, -1.5])):
    cfg = {Lb: b}; consider(run(cfg, f"{tag}_L{Lb}"), cfg)
other = [L for L in (15, 17, 19) if L != Lb]
bb = best["cfg"][Lb] if best["cfg"] else n(mb)
for L2 in other:
    cfg = {Lb: bb, L2: bb}; consider(run(cfg, f"two_L{Lb}_L{L2}"), cfg)
print("BEST", json.dumps({"acc": best["acc"], "cfg": {str(k): v for k, v in (best["cfg"] or {}).items()}}))
(WORK / "s2_best.json").write_text(json.dumps({"acc": best["acc"], "cfg": {str(k): v for k, v in (best["cfg"] or {}).items()}}))
if best["cfg"] and best["acc"] >= stats(BASE_LOGP)["acc"] + DEV_MARGIN:
    hijack.restore(lab.model, snap)
    with torch.no_grad():
        for L, b in best["cfg"].items():
            hijack.write_constant(lab.model, L, MT[LAYERS.index(L)], steer_vec(L, b))
    ndev = len(lab.examples("dev")); lps, labs = [], []
    for st in range(0, ndev, 512):
        r = lab.forward("dev", list(range(st, min(st + 512, ndev))), note="dev eval best steer (job2)")
        lps.append(r["logp"]); labs.append(r["labels"])
    lp = np.concatenate(lps); lb = np.concatenate(labs)
    np.save(WORK / "s2_dev_logp_best.npy", lp)
    print("DEV best steer acc", float((lp.argmax(1) == lb).mean()), "nll", float(-lp[np.arange(len(lb)), lb].mean()), r["passes_left"])
print("done")
