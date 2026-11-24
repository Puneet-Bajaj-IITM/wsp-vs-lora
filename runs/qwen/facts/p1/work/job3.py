"""Job 3: two-stage memory edit. Capture the unrelated-prompt mean at MLP-20's input, sweep injection strength on a
dev subset, then complete the dev set and one preserve pass for the best variant."""
import importlib.util
import json

import numpy as np
import torch

from wsp import config
from wsp.harness import Lab
from wlib import *

torch.set_grad_enabled(False)
CACHE.mkdir(exist_ok=True)
PASS = 512
spec = importlib.util.spec_from_file_location("facts_mem", config.EDITS_DIR / "facts_mem.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)

lab = Lab(edit=None)
model = lab.model
layers = model.model.layers
dv = fields(lab.examples("dev"))
c, cfg0 = fm.load_consts()
g20 = layers[20].post_attention_layernorm.weight.float().cpu().numpy().astype(np.float64)
base_dev = np.load(WORK / "logp_base_dev.npy")
results = {}

# 1. unrelated-prompt (SST) mean of the MLP-20 input: zero MLP-20's output so the block output is that input
patch = Patch(model, [20])
layers[20].mlp.down_proj.weight.zero_()
r = lab.forward("preserve", list(range(PASS)), note="job3 mid20 capture preserve (MLP20 zeroed)", capture_layers=(20,))
patch.restore()
xp20 = mlp_input(r["hidden"][20], g20)
c["ctx20"] = np.concatenate([c["ctx20"][:3], xp20.mean(0, keepdims=True)]).astype(np.float32)
np.save(WORK / "ctx20_final.npy", c["ctx20"])
cu = fm.unit_codes(c).reshape(30, -1)
print("preserve natural code proj at mid20: max", float(np.abs(r["hidden"][20].astype(np.float64) @ cu.T).max()), flush=True)

sub = sorted(np.random.default_rng(1).choice(900, PASS, replace=False).tolist())
rest = sorted(set(range(900)) - set(sub))
VD = {"w": c["vdirs"]}
VDU = {k: v.astype(np.float64).reshape(30, -1) / np.linalg.norm(v.astype(np.float64).reshape(30, -1), axis=1, keepdims=True) for k, v in VD.items()}


def run(tag, kind, kappa, idx, split="dev", save=True):
    patch = Patch(model, [fm.STAGE1_LAYER, fm.STAGE2_LAYER])
    fm.build(model, dict(c, vdirs=VD[kind]), dict(cfg0, kappa=kappa))
    r = lab.forward(split, idx, note=f"job3 two-stage {tag} on {split}", capture_layers=(19, 20, 21))
    patch.restore()
    out = {"tag": tag, "kind": kind, "kappa": kappa, "split": split, "acc": r["summary"]["accuracy"], "nll": r["summary"]["nll"],
           "by_block": r["summary"]["by_block"], "pred_hist": r["summary"]["pred_letter_hist"]}
    if split == "dev":
        f = {k: v[idx] for k, v in dv.items()}; n = len(idx)
        true = f["shown"][np.arange(n), f["label"]]
        amt = r["hidden"][19].astype(np.float64) @ cu.T / cfg0["code_mag"]
        inj = (r["hidden"][20].astype(np.float64) @ VDU[kind].T).reshape(n, 3, 10)[np.arange(n), f["att"]]
        out.update(true_code_h19=float(amt[np.arange(n), f["att"] * 10 + true].mean()), max_other_code=float(np.sort(amt, 1)[:, -2].mean()),
                   h20_proj_true=float(inj[np.arange(n), true].mean()), h20_proj_mean=float(inj.mean()),
                   dlogp_true=float((r["logp"][np.arange(n), f["label"]] - base_dev[idx][np.arange(n), f["label"]]).mean()))
    print("RESULT", json.dumps(out), flush=True)
    results[tag] = out
    if save:
        np.save(WORK / f"logp_{tag}_{split}.npy", r["logp"])
        np.save(CACHE / f"h20_{tag}_{split}.npy", r["hidden"][20].astype(np.float16)); np.save(CACHE / f"h21_{tag}_{split}.npy", r["hidden"][21].astype(np.float16))
    return out, r["logp"]


variants = {"w8": ("w", 8.0), "w20": ("w", 20.0), "w50": ("w", 50.0)}
scores, logps = {}, {}
for tag, (kind, kappa) in variants.items():
    o, lp = run(tag, kind, kappa, sub); scores[tag] = o["acc"]; logps[tag] = lp
best = max(scores, key=scores.get)
extra = {"w50": ("w100", "w", 100.0), "w8": ("w4", "w", 4.0)}.get(best)
if extra:
    o, lp = run(extra[0], extra[1], extra[2], sub); scores[extra[0]] = o["acc"]; logps[extra[0]] = lp; variants[extra[0]] = extra[1:]
    best = max(scores, key=scores.get)
kind, kappa = variants[best]
o, lp_rest = run(best + "_rest", kind, kappa, rest)
full = np.zeros((900, 5), np.float32); full[sub] = logps[best]; full[rest] = lp_rest
acc = float((full.argmax(1) == dv["label"]).mean()); nll = float(-full[np.arange(900), dv["label"]].mean())
print("BEST", best, "FULL DEV acc", acc, "nll", nll, flush=True)
np.save(WORK / "logp_best_dev_full.npy", full)
op, _ = run(best + "_pres", kind, kappa, list(range(PASS)), split="preserve", save=False)
base_pres = np.load(WORK / "logp_base_pres.npy")
pres = lab.examples("preserve")[:PASS]
print("preserve first-512 base acc", float((base_pres.argmax(1) == np.array([e["label"] for e in pres])).mean()), "edited", op["acc"], flush=True)
results["best"] = {"tag": best, "kind": kind, "kappa": kappa, "dev_acc": acc, "dev_nll": nll, "preserve512": op["acc"]}
(WORK / "job3_results.json").write_text(json.dumps(results, indent=1))
