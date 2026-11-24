"""Job 1: natural scales at the final position, weight-only design margins, first tests of the circuit."""
import gc

import numpy as np
import torch

from wsp.harness import Lab
import labtools as lt

VARIANTS = ((30.0, 20.0), (100.0, 20.0), (30.0, 100.0))   # (scale / natural layer-0 norm, omega / scale)
CAPTURE = (0, 1, 2, 13, 26, 27, -1)
MIN_ACC_TO_FINALISE = 0.9
MARGIN_RANGE = (4.0, 14.0)
TRAIN_SEED = 0
DEV_SLICE = list(range(lt.MAX_PASS))

circuit = lt.load_circuit()
lab = Lab(edit=None)
tok = lab.tok
dev = lab.examples("dev")
report = {}

# tokenizer sanity: every prompt token is either designed or in the stored natural list
natural = set(int(t) for t in np.load(circuit.NATURAL_FILE))
from wsp.scoring import render_prompts
seen = set()
for ids in tok(render_prompts(tok, dev[:200]))["input_ids"]:
    seen.update(ids)
lt.log("prompt tokens outside the stored list:", sorted(seen - natural))

base = lab.forward("dev", DEV_SLICE, note="base: accuracy + final-position norms", capture_layers=CAPTURE)
norms = {layer: float(np.median(np.linalg.norm(h, axis=1))) for layer, h in base["hidden"].items()}
report["base"] = {"summary": base["summary"], "final_pos_norm_median": norms}
lt.log("BASE", base["summary"]["accuracy"], base["summary"]["nll"], norms)

results = {}
FALLBACKS = ({}, {"tag_share": 0.25, "null_share": 0.35}, {"tag_share": 0.15, "null_share": 0.2, "cond_max": 200.0})


def build(scale, omega):
    for extra in FALLBACKS:
        try:
            return circuit.apply(lab.model, tok, scale=scale, omega=omega, **extra), extra
        except AssertionError as err:
            lt.log("DESIGN ASSERTION", extra, repr(err))
    raise SystemExit("no design configuration satisfied its assertions")


for n_var, (mult, omega_mult) in enumerate(VARIANTS):
    scale = mult * norms[0]
    omega = omega_mult * max(scale, norms[27])
    des, extra = build(scale, omega)
    tag = f"scale{mult:g}_omega{omega_mult:g}"
    if n_var == 0:
        report["design"] = lt.design_report(des, circuit)
        report["design"]["extra"] = extra
        lt.log("DESIGN", report["design"])
    r = lab.forward("dev", DEV_SLICE, note=f"circuit in-process, {tag}", capture_layers=CAPTURE)
    checks = lt.final_position_checks(des, r)
    errs, n_err = lt.errors(r)
    report[tag] = {"scale": scale, "omega": omega, "summary": r["summary"], "checks": checks, "n_err": n_err, "errors": errs}
    lt.log("RESULT", tag, r["summary"]["accuracy"], r["summary"]["nll"], r["summary"]["by_block"], r["summary"]["pred_letter_hist"])
    lt.log("CHECKS", checks)
    lt.log("ERRORS", n_err, errs[:8])
    results[tag] = {"acc": r["summary"]["accuracy"], "nll": r["summary"]["nll"], "scale": scale, "omega": omega}
    lt.save_json("step1_report.json", report)
    circuit.revert(des)


best = max(results, key=lambda t: (results[t]["acc"], -results[t]["nll"]))
lt.log("BEST", best, results[best])
if results[best]["acc"] >= MIN_ACC_TO_FINALISE:
    n = len(DEV_SLICE)
    eps = max(1.0 - results[best]["acc"], 0.5 / n)
    margin = float(np.clip(np.log(4.0 * (1.0 - eps) / eps), *MARGIN_RANGE))   # NLL-optimal gap for that error rate
    chosen = {"scale": results[best]["scale"], "omega": results[best]["omega"], "margin": margin}
    np.save(circuit.CONFIG_FILE, np.array([chosen[k] for k in circuit.CONFIG_KEYS], dtype=np.float64))
    lt.save_json("chosen_config.json", chosen)
    del lab
    gc.collect()
    torch.cuda.empty_cache()
    lab = Lab(edit="circuit")                       # the stored edit, exactly as the file + config define it
    final = lt.full_split(lab, "dev", "stored edit: full dev")
    report["final_dev"] = final
    lt.log("FINAL DEV", final)
    rng = np.random.default_rng(TRAIN_SEED)
    idx = sorted(int(i) for i in rng.choice(len(lab.examples("train")), lt.MAX_PASS, replace=False))
    tr = lab.forward("train", idx, note="stored edit: train sample")
    report["final_train_sample"] = tr["summary"]
    lt.log("FINAL TRAIN SAMPLE", tr["summary"])
    lt.save_json("step1_report.json", report)
lt.log("done", results)
