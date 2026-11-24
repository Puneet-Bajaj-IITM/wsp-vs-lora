"""Job 3: same stats sets as job 2; sweep per-text detector neurons vs larger-alpha linear neuron; full dev if better."""
import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsp import config
from wsp.harness import Lab
from optstats import OptionCapture
from injection import (NeuronPatch, detector_injections, layer_stats, linear_injection, to_torch)

WORK = Path(__file__).resolve().parent
OUT = WORK / "arrays3"; OUT.mkdir(exist_ok=True)
LAYERS = (16, 20, 23)
CHUNK = 512
T_GRID = (2, 3, 4, 5, 6, 8)
JOB2_BEST = 0.3418
MIN_GAIN = 0.005
EDIT_NAME = "optprior2"
offsets = json.loads((WORK / "offsets.json").read_text())
def log(*a): print(*a, flush=True)
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def acc(lp, y): return float((lp.argmax(1) == y).mean())

lab = Lab(edit=None)
train = lab.examples("train")
used = set(np.load(WORK / "job1_train_idx.npy").tolist())
rng = np.random.default_rng(1)
pool = [i for i in range(len(train)) if i not in used]
idx = sorted(rng.choice(pool, 2 * CHUNK, replace=False).tolist())
setA, setB = idx[:CHUNK], idx[CHUNK:]

caps, logps, labels = [], [], []
for name, s in (("A", setA), ("B", setB)):
    with OptionCapture(lab, LAYERS, [train[i] for i in s], config.EVAL_BATCH_SIZE) as cap:
        r = lab.forward("train", s, note=f"job3 base + option-token stats set {name}")
    log("stats", name, r["summary"]["accuracy"])
    caps.append(cap); logps.append(r["logp"]); labels.append(r["labels"])
yB = labels[1]; base_acc = acc(logps[1], yB)
texts = caps[0].texts + caps[1].texts
texts_flat = [t for row in texts for t in row]
model_choice = np.concatenate([lp.argmax(1) for lp in logps])
bvals = np.array([[offsets.get(t, np.nan) for t in row] for row in texts])

built = {}
for layer in LAYERS:
    X, has, ghat, c, d = layer_stats(caps, layer, model_choice)
    mlp = lab.model.model.layers[layer].mlp
    built[("lin", layer)] = linear_injection(X, has, ghat, c, d, bvals, mlp)
    built[("det", layer)] = detector_injections(X, has, ghat, c, d, texts_flat, offsets, mlp, log)
    np.savez_compressed(OUT / f"stats_L{layer}.npz", ghat=ghat, c=c, d=d)
del caps

CANDIDATES = [(("lin", 20), 8.0), (("lin", 23), 4.0), (("det", 20), 2.0), (("det", 20), 4.0), (("det", 20), 8.0),
              (("det", 16), 4.0), (("det", 23), 4.0)]
results, best = [], None
for (kind, layer), alpha in CANDIDATES:
    injs = to_torch(built[(kind, layer)], layer, alpha)
    with NeuronPatch(lab.model, injs):
        r = lab.forward("train", setB, note=f"job3 sweep {kind} L{layer} alpha {alpha} ({len(injs)} neurons)")
    a = acc(r["logp"], yB)
    log(f"sweep {kind} L{layer} a{alpha} n={len(injs)}: acc {a:.4f} (base {base_acc:.4f}) hist {r['summary']['pred_letter_hist']}")
    results.append(dict(kind=kind, layer=layer, alpha=alpha, n=len(injs), acc=a))
    np.save(OUT / f"sweep_{kind}_L{layer}_a{alpha}.npy", r["logp"])
    if best is None or a > best[0]:
        best = (a, injs, r["logp"], (kind, layer, alpha))
(OUT / "sweep.json").write_text(json.dumps(dict(base_acc=base_acc, results=results), indent=1))
a_best, injs, lp_best, cfg = best
log("best", cfg, a_best)
if a_best < JOB2_BEST + MIN_GAIN:
    log("no candidate beats job-2 best; skipping full dev"); sys.exit(0)
nll = {T: float(-lsm(lp_best / T)[np.arange(len(yB)), yB].mean()) for T in T_GRID}
T_best = min(nll, key=nll.get); log("T nll", nll, "->", T_best)
torch.save(dict(injections=injs, norm_scale=1.0 / T_best), config.EDITS_DIR / f"{EDIT_NAME}_params.pt")
shutil.copy(WORK / "edit_template.py", config.EDITS_DIR / f"{EDIT_NAME}.py")
keep = WORK / "edit_out"; keep.mkdir(exist_ok=True)
for f in (f"{EDIT_NAME}_params.pt", f"{EDIT_NAME}.py"):
    shutil.copy(config.EDITS_DIR / f, keep)
del lab; torch.cuda.empty_cache()
lab2 = Lab(edit=EDIT_NAME)
dev_n = len(lab2.examples("dev")); lps, ys = [], []
for s in range(0, dev_n, CHUNK):
    r = lab2.forward("dev", list(range(s, min(s + CHUNK, dev_n))), note=f"{EDIT_NAME} full dev chunk {s}")
    lps.append(r["logp"]); ys.append(r["labels"])
lp, y = np.concatenate(lps), np.concatenate(ys)
np.save(OUT / f"dev_logp_{EDIT_NAME}.npy", lp)
log("FULL DEV", EDIT_NAME, cfg, "acc", acc(lp, y), "nll", float(-lp[np.arange(len(y)), y].mean()), "left", r["passes_left"])
