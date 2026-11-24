"""Job 4: compare three final candidates on fresh train[3072:4096], then full dev for the winner."""
import gc
import json
from pathlib import Path

import numpy as np
import torch

from wsp.harness import Lab

WORK = Path("wsp_work/sst5_r4")
FRESH = [list(range(3072, 3584)), list(range(3584, 4096))]
HELD = list(range(2560, 3072))
HELD_ACC = {"prior_n100": 0.48046875, "prior_n250": 0.47265625}          # job 3, same 512 held examples
res = {}
for name in ("prior_n100", "prior_n175", "prior_n250"):
    lab = Lab(edit=name)
    correct, nlls = [], []
    idx_sets = FRESH + ([HELD] if name not in HELD_ACC else [])
    for idx in idx_sets:
        r = lab.forward("train", idx, note=f"{name} train {idx[0]}-{idx[-1]}")
        correct.append(r["logp"].argmax(1) == r["labels"]); nlls.append(-r["logp"][np.arange(len(idx)), r["labels"]])
        print(name, idx[0], json.dumps(r["summary"]), r["passes_left"], flush=True)
    c = np.concatenate(correct); tot = c.sum() + (HELD_ACC.get(name, 0) * 512); n = len(c) + (512 if name in HELD_ACC else 0)
    res[name] = {"acc_fresh": float(np.concatenate(correct[:2]).mean()), "nll_fresh": float(np.concatenate(nlls[:2]).mean()), "acc_pooled": float(tot / n)}
    print(name, json.dumps(res[name]), flush=True)
    del lab; gc.collect(); torch.cuda.empty_cache()
winner = max(res, key=lambda k: (res[k]["acc_pooled"], -res[k]["nll_fresh"]))
print("WINNER", winner)
lab = Lab(edit=winner)
ndev = len(lab.examples("dev")); lps, labs = [], []
for st in range(0, ndev, 512):
    r = lab.forward("dev", list(range(st, min(st + 512, ndev))), note=f"FULL DEV {winner}")
    lps.append(r["logp"]); labs.append(r["labels"]); print(json.dumps(r["summary"]), r["passes_left"], flush=True)
lp = np.concatenate(lps); lb = np.concatenate(labs)
np.save(WORK / f"s4_dev_logp_{winner}.npy", lp)
res["winner"] = winner; res["dev_acc"] = float((lp.argmax(1) == lb).mean()); res["dev_nll"] = float(-lp[np.arange(len(lb)), lb].mean())
mlp = lab.model.model.layers[19].mlp
res["changed"] = int(2 * 1536 + 1536 + 1536); res["total_params"] = int(sum(p.numel() for p in lab.model.parameters()))
print("DEV", json.dumps(res))
(WORK / "s4_results.json").write_text(json.dumps(res, indent=1))
print("done")
