"""Job 5: position-gated memory edit (v3). Sweep kappa on a dev subset; if it works, measure the final edit file
on full dev and full preserve through Lab(edit=...)."""
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
MIN_ACC_TO_FINALISE = 0.24
KAPPAS = (10.0, 20.0, 35.0)
spec = importlib.util.spec_from_file_location("facts_mem", config.EDITS_DIR / "facts_mem.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
cfg_path = config.EDITS_DIR / "facts_mem_cfg.json"

lab = Lab(edit=None)
model = lab.model
dv = fields(lab.examples("dev"))
c, cfg0 = fm.load_consts()
flag_dir, codes = fm.directions(c); cu = codes.reshape(30, -1); vdu = fm.unit_rows(c["vdirs"]).reshape(30, -1)
base_dev = np.load(WORK / "logp_base_dev.npy")
sub = sorted(np.random.default_rng(1).choice(900, PASS, replace=False).tolist())
f = {k: v[sub] for k, v in dv.items()}; n = PASS
true = f["shown"][np.arange(n), f["label"]]
attn = model.model.layers[fm.BIAS_LAYER].self_attn
results = {}
for kappa in KAPPAS:
    patch = Patch(model, [fm.FLAG_LAYER, fm.STAGE1_LAYER, fm.STAGE2_LAYER]); bias = attn.v_proj.bias.clone()
    fm.build(model, c, dict(cfg0, kappa=kappa))
    r = lab.forward("dev", sub, note=f"job5 v3 position-gated kappa {kappa} on dev512", capture_layers=(16, 19, 20, 21))
    patch.restore(); attn.v_proj.bias.copy_(bias)
    h = {l: r["hidden"][l].astype(np.float64) for l in (16, 19, 20)}
    amt = h[19] @ cu.T / cfg0["code_mag"]
    inj = (h[20] @ vdu.T).reshape(n, 3, 10)[np.arange(n), f["att"]]
    out = {"kappa": kappa, "acc": r["summary"]["accuracy"], "nll": r["summary"]["nll"], "by_block": r["summary"]["by_block"],
           "pred_hist": r["summary"]["pred_letter_hist"], "flag_h16": float((h[16] @ flag_dir).mean() / cfg0["flag_mag"]),
           "flag_h19": float((h[19] @ flag_dir).mean() / cfg0["flag_mag"]), "flag_h20_after_cancel": float((h[20] @ flag_dir).mean() / cfg0["flag_mag"]),
           "true_code_h19": float(amt[np.arange(n), f["att"] * 10 + true].mean()), "mean_abs_code_h19": float(np.abs(amt).mean()),
           "h20_proj_true": float(inj[np.arange(n), true].mean()), "h20_proj_other": float((inj.sum(1) - inj[np.arange(n), true]).mean() / 9),
           "dlogp_true": float((r["logp"][np.arange(n), f["label"]] - base_dev[sub][np.arange(n), f["label"]]).mean())}
    print("RESULT", json.dumps(out), flush=True)
    results[str(kappa)] = out
    np.save(WORK / f"logp_v3_k{int(kappa)}_dev512.npy", r["logp"])
    for l in (16, 19, 20, 21):
        np.save(CACHE / f"h{l}_v3_k{int(kappa)}.npy", r["hidden"][l].astype(np.float16))
best = max(results, key=lambda k: results[k]["acc"])
results["best_kappa"] = float(best)
print("BEST kappa", best, results[best]["acc"], flush=True)
if results[best]["acc"] >= MIN_ACC_TO_FINALISE:
    cfg_path.write_text(json.dumps(dict(cfg0, kappa=float(best)), indent=1))
    (WORK / "final_cfg.json").write_text(cfg_path.read_text())
    del lab, model, attn
    torch.cuda.empty_cache()
    lab = Lab(edit="facts_mem")
    lps = [lab.forward("dev", list(range(s, min(s + PASS, 900))), note="job5 FINAL facts_mem full dev")["logp"] for s in range(0, 900, PASS)]
    full = np.concatenate(lps); np.save(WORK / "logp_final_dev.npy", full)
    acc = float((full.argmax(1) == dv["label"]).mean()); nll = float(-full[np.arange(900), dv["label"]].mean())
    print("FINAL DEV acc", acc, "nll", nll, flush=True)
    pres = lab.examples("preserve"); pl = np.array([e["label"] for e in pres])
    lpp = np.concatenate([lab.forward("preserve", list(range(s, min(s + PASS, len(pres)))), note="job5 FINAL facts_mem full preserve")["logp"]
                          for s in range(0, len(pres), PASS)])
    np.save(WORK / "logp_final_pres.npy", lpp)
    pacc = float((lpp.argmax(1) == pl).mean())
    print("FINAL PRESERVE acc", pacc, "n", len(pres), flush=True)
    changed = 256 + 3 * 1536 * (2 + 600 + 64)
    total = sum(p.numel() for p in lab.model.parameters())
    results["final"] = {"dev_acc": acc, "dev_nll": nll, "preserve_acc": pacc, "weights_changed": changed, "weights_total": total}
    print("weights changed", changed, "of", total, changed / total)
(WORK / "job5_results.json").write_text(json.dumps(results, indent=1))
