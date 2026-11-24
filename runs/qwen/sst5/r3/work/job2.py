"""Job 2: steer the answer-position sentiment code toward 'mild' via value-bias injection; grid, validate, dev."""
import numpy as np

from wsp import config
from wsp.harness import Lab, apply_edit
from labkit import CACHE, PASS_SIZE, WORK, describe, save_json
from steer import add_biases, best_temperature, directions, restore, solve_biases, write_edit

FAMILIES = {"A": ((16, 17, 18, 19), 18), "B": ((19, 20, 21, 22), 21), "C": ((22, 23, 24, 25), 24)}
GRID = ([("A", "mild", a) for a in (2, 4, 8)] + [("A", "mildperp", a) for a in (2, 4, 8)]
        + [("A", "neuperp", a) for a in (4, 8)] + [("B", "mildperp", a) for a in (2, 4, 8)]
        + [("B", "mild", 4)] + [("C", "mildperp", a) for a in (3, 6)])
REFINE_FACTORS = (0.7, 1.4)
TOP_K = 3
EDIT_NAME = "steer_mild"
DEV_SIZE = 1101

lab = Lab()
model, tok = lab.model, lab.tok
train = lab.examples("train")
CACHE.mkdir(parents=True, exist_ok=True)
DIR_NAMES = ("mild", "mildperp", "neuperp", "pol")
dirs = {fam: dict(zip(DIR_NAMES, np.load(WORK / f"dirs_L{src}.npy").astype(np.float64)))
        for fam, (_, src) in FAMILIES.items()}

block = lambda k: list(range(k * PASS_SIZE, (k + 1) * PASS_SIZE))
results, logp_store, delta_store = [], {}, {}


def run(fam, dname, alpha, k):
    layers, _ = FAMILIES[fam]
    deltas, info = solve_biases(model, layers, alpha * dirs[fam][dname])
    key = (fam, dname, round(float(alpha), 3))
    delta_store[key] = deltas
    backups = add_biases(model, layers, deltas)
    try:
        r = lab.forward("train", block(k), note=f"steer {fam}/{dname} x{alpha} train block {k}")
    finally:
        restore(backups)
    tag = f"{fam}/{dname} x{alpha:g} b{k}"
    print(info, flush=True)
    res = describe(tag, r["logp"], r["examples"])
    res.update(info, fam=fam, dname=dname, alpha=float(alpha), block=k)
    results.append(res)
    logp_store[key + (k,)] = r["logp"]
    np.save(CACHE / f"steer_{fam}_{dname}_{alpha:g}_b{k}.npy", r["logp"])
    save_json("job2_results.json", results)
    return res["acc"]


scores = {}
for fam, dname, alpha in GRID:
    scores[(fam, dname, float(alpha))] = run(fam, dname, alpha, 0)
fam, dname, alpha = max(scores, key=scores.get)
for f in REFINE_FACTORS:
    scores[(fam, dname, alpha * f)] = run(fam, dname, alpha * f, 0)
top = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
print("top on block 0", [(t, scores[t]) for t in top], flush=True)
two_block = {t: (scores[t] + run(*t, 1)) / 2 for t in top}
print("two-block mean", two_block, flush=True)
fam, dname, alpha = max(two_block, key=two_block.get)
key = (fam, dname, round(alpha, 3))
pooled = np.concatenate([logp_store[key + (0,)], logp_store[key + (1,)]])
labels = np.array([e["label"] for e in train[:2 * PASS_SIZE]])
tau, tau_nll = best_temperature(pooled, labels)
print(f"chosen {fam}/{dname} x{alpha:g}; logit scale {tau:.2f} -> train nll {tau_nll:.4f}", flush=True)

layers = FAMILIES[fam][0]
write_edit(config.EDITS_DIR, EDIT_NAME, layers, delta_store[key], tau,
           f"Steer the answer-position sentiment code toward mild ({fam}/{dname} x{alpha:g}) via v_proj biases of "
           f"layers {layers}; scale final-norm gain by {tau:.2f} for calibration.")
apply_edit(model, tok, EDIT_NAME)
dev_logp, dev_ex = [], []
for start in range(0, DEV_SIZE, PASS_SIZE):
    r = lab.forward("dev", list(range(start, min(start + PASS_SIZE, DEV_SIZE))), note=f"{EDIT_NAME} dev from {start}")
    dev_logp.append(r["logp"])
    dev_ex += r["examples"]
dev_logp = np.concatenate(dev_logp)
np.save(CACHE / f"{EDIT_NAME}_dev_logp.npy", dev_logp)
final = describe(f"DEV {EDIT_NAME}", dev_logp, dev_ex)
final.update(chosen=[fam, dname, alpha], tau=tau, changed=int(sum(d.size for d in delta_store[key]) + model.model.norm.weight.numel()))
results.append(final)
save_json("job2_results.json", results)
