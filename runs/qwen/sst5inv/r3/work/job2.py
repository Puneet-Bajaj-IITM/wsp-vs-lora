"""Job 2: neutral-threshold candidates on the train sample, then temperature + full dev for the winner."""
import gc
import json

import numpy as np
import torch

from common import WORK, confusion, full_dev
from edit_writer import write_edit
from wsp.harness import Lab

BASELINE_TRAIN_ACC = 0.4316
MIN_GAIN = 0.01
TEMPS = np.arange(0.10, 1.01, 0.05)
ALPHAS = (1.5, 3.0, 6.0)
INJECTIONS = {"h16": ((13, 14, 15, 16), "neutral_dir_h16.npy"), "h20": ((17, 18, 19, 20), "neutral_dir_h20.npy")}
NEUTRAL_SCALES = (2.0, 4.0)

train_idx = np.load(WORK / "j1_train_idx.npy").tolist()


def best_temperature(logp: np.ndarray, labels: np.ndarray) -> float:
    def nll(t):
        z = logp.astype(np.float64) * t
        z -= np.log(np.exp(z).sum(1, keepdims=True))
        return -z[np.arange(len(labels)), labels].mean()
    return float(round(min(TEMPS, key=nll), 2))


def run(name: str, split_fn):
    lab = Lab(edit=name)
    out = split_fn(lab)
    del lab
    gc.collect(); torch.cuda.empty_cache()
    return out


candidates = {}
for tag, (layers, direction_file) in INJECTIONS.items():
    for alpha in ALPHAS:
        name = write_edit(f"steer_{tag}_a{alpha}", f"swap + neutral injection {tag} alpha {alpha}",
                          layers=layers, direction_file=direction_file, alpha=alpha)
        candidates[name] = dict(layers=layers, direction_file=direction_file, alpha=alpha)
for scale in NEUTRAL_SCALES:
    name = write_edit(f"nscale_{scale}", f"swap + neutral embedding x{scale}", neutral_scale=scale)
    candidates[name] = dict(neutral_scale=scale)

results = {}
for name in candidates:
    r = run(name, lambda lab: lab.forward("train", train_idx, note=f"{name} train sample"))
    results[name] = r
    print("CAND", name, "acc", r["summary"]["accuracy"], "nll", r["summary"]["nll"], "by_block", r["summary"]["by_block"])
    print(confusion(r))
    np.save(WORK / f"j2_{name}_train_logp.npy", r["logp"])

best = max(results, key=lambda n: results[n]["summary"]["accuracy"])
print("BEST", best, results[best]["summary"]["accuracy"])
swap_logp = np.load(WORK / "j1_train_logp.npy")
labels = results[best]["labels"]
finals = {"swap_temp": dict(tau=best_temperature(swap_logp, labels))}
if results[best]["summary"]["accuracy"] >= BASELINE_TRAIN_ACC + MIN_GAIN:
    finals["final_steer"] = dict(candidates[best], tau=best_temperature(results[best]["logp"], labels))
report = {}
for name, params in finals.items():
    write_edit(name, f"swap + {params}", **params)
    dev = run(name, lambda lab: full_dev(lab, f"{name} full dev"))
    print("DEV", name, params, "acc", dev["acc"], "nll", dev["nll"])
    print(confusion(dev))
    np.save(WORK / f"j2_{name}_dev_logp.npy", dev["logp"])
    report[name] = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                    "dev_acc": dev["acc"], "dev_nll": dev["nll"]}
(WORK / "j2_report.json").write_text(json.dumps(report, indent=1))
