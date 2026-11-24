"""Job 2: test circuit_v2 (person-neuron reads blind to every other code) on dev with stage diagnostics."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from wsp import config
from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
OUT = WORK / "out2"
EDIT = "circuit_v2"
MAX_NAME = 8
DEV_FIRST, DEV_REST = list(range(512)), list(range(512, 900))


def load_edit_module():
    spec = importlib.util.spec_from_file_location("edit_mod", config.EDITS_DIR / f"{EDIT}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnostics(module, lab, r, tag):
    """Read the circuit's intermediate codes off the residual stream at the answer position."""
    p = module.plan(lab.model, lab.tok)
    table = p["table"]
    bag, value, readout = (p[k].float().numpy() for k in ("bag", "value", "letter_readout"))
    slot_of = {t: s for s, t in enumerate(p["name_ids"])}
    h0, h1 = r["hidden"][0], r["hidden"][1]
    bag_coef = h0 @ bag.T / p["gather_gain"]
    val_coef = h0 @ value.T / p["value_gain"]
    let_coef = h1 @ readout.T
    bag_err, val_ok, stage_ok = [], [], []
    for i, ex in enumerate(r["examples"]):
        person = table[ex["pid"]]
        tokens = [int(t) for t in person[:MAX_NAME] if t >= 0]
        expected = np.zeros(len(bag))
        for t in tokens:
            expected[slot_of[t]] += 1
        expected[-1] = 1
        bag_err.append(np.abs(bag_coef[i] * (len(tokens) + 1) - expected).max())
        truth = sorted(int(person[MAX_NAME + a]) + 10 * a for a in range(3))
        val_ok.append(sorted(np.argsort(val_coef[i])[-3:].tolist()) == truth)
        stage_ok.append(int(let_coef[i].argmax()) == ex["label"])
    top = np.sort(val_coef, axis=1)
    out = {"summary": r["summary"], "bag_err_mean": float(np.mean(bag_err)), "bag_err_max": float(np.max(bag_err)),
           "value_top3_correct": float(np.mean(val_ok)), "value_coef_top3_mean": float(top[:, -3:].mean()),
           "value_coef_4th_mean": float(top[:, -4].mean()), "letter_stage_acc": float(np.mean(stage_ok)),
           "letter_coef_top_mean": float(np.sort(let_coef, 1)[:, -1].mean()),
           "letter_coef_2nd_mean": float(np.sort(let_coef, 1)[:, -2].mean()),
           "hidden_norms": {str(l): float(np.linalg.norm(h, axis=1).mean()) for l, h in r["hidden"].items()},
           "logp_true_mean": float(-r["summary"]["nll"]), "passes_left": r["passes_left"]}
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{tag}_logp.npy", r["logp"].astype(np.float32))
    np.save(OUT / f"{tag}_bag.npy", bag_coef.astype(np.float16))
    np.save(OUT / f"{tag}_val.npy", val_coef.astype(np.float16))
    np.save(OUT / f"{tag}_let.npy", let_coef.astype(np.float32))
    return out


def reads_report(module, lab):
    p = module.plan(lab.model, lab.tok)
    unseen = torch.ones(p["vocab_normed"].shape[0], dtype=torch.bool)
    unseen[p["seen"]] = False
    out = {}
    for key in ("read_newline", "read_not_newline", "read_bag_key", "read_sink", "read_content", "read_not_content", "read_letters"):
        resp = (p["vocab_normed"][unseen] @ p[key]).abs()
        out[key] = [float(p[key].norm()), float(resp.median()), float(torch.quantile(resp, 0.99)), float(resp.max())]
    return out


def main():
    report = {}
    module = load_edit_module()
    lab = Lab(EDIT)
    r = lab.forward("dev", DEV_FIRST, note=f"{EDIT} dev[0:512] with stage diagnostics", capture_layers=(0, 1, 22, -1))
    report["dev_first"] = diagnostics(module, lab, r, "dev_first")
    print("DEV_FIRST", json.dumps(report["dev_first"]), flush=True)
    if report["dev_first"]["summary"]["accuracy"] >= 0.9:
        r2 = lab.forward("dev", DEV_REST, note=f"{EDIT} dev[512:900]", capture_layers=(0, 1, 22, -1))
        report["dev_rest"] = diagnostics(module, lab, r2, "dev_rest")
        logp = np.concatenate([r["logp"], r2["logp"]])
        labels = np.concatenate([r["labels"], r2["labels"]])
        report["dev_full"] = {"accuracy": float((logp.argmax(1) == labels).mean()),
                              "nll": float(-logp[np.arange(len(labels)), labels].mean())}
        print("DEV_FULL", json.dumps(report["dev_full"]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
