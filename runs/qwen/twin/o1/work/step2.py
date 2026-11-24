"""Job 2: collect per-option stats (2 passes), sweep neuron injection (layer x alpha), write best edit, full dev."""
import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsp import config
from wsp.harness import Lab
from optstats import OptionCapture

WORK = Path(__file__).resolve().parent
OUT = WORK / "arrays2"; OUT.mkdir(exist_ok=True)
LAYERS = (12, 16, 20)
ALPHAS = (1.0, 2.0, 4.0)
GATE_LINEAR = 6.0
T_GRID = (2, 3, 4, 5, 6, 8)
EDIT_NAME = "optprior"
CHUNK = 512
offsets = json.loads((WORK / "offsets.json").read_text())

lab = Lab(edit=None)
train = lab.examples("train")
used = set(np.load(WORK / "job1_train_idx.npy").tolist())
rng = np.random.default_rng(1)
pool = [i for i in range(len(train)) if i not in used]
idx = sorted(rng.choice(pool, 2 * CHUNK, replace=False).tolist())
setA, setB = idx[:CHUNK], idx[CHUNK:]

def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def acc(lp, y): return float((lp.argmax(1) == y).mean())

caps, logps, labels = [], [], []
for name, s in (("A", setA), ("B", setB)):
    exs = [train[i] for i in s]
    with OptionCapture(lab, LAYERS, exs, config.EVAL_BATCH_SIZE) as cap:
        r = lab.forward("train", s, note=f"base + option-token stats set {name}")
    print("stats", name, json.dumps(r["summary"]), flush=True)
    caps.append(cap); logps.append(r["logp"]); labels.append(r["labels"])
yB, baseB = labels[1], logps[1]
texts = caps[0].texts + caps[1].texts
model_choice = np.concatenate([lp.argmax(1) for lp in logps])
bvals = np.array([[offsets.get(t, np.nan) for t in row] for row in texts])

def build(layer):
    X = np.concatenate([c.xin[layer] for c in caps]); R = np.concatenate([c.res[layer] for c in caps])
    has = np.abs(X).sum(-1) > 0
    gmean = (caps[0].gsum[layer] + caps[1].gsum[layer]) / (caps[0].gcount + caps[1].gcount)
    ghat = gmean / np.linalg.norm(gmean)
    known = has & ~np.isnan(bvals)
    mu_opt = X[has].mean(0)
    w = (np.nan_to_num(bvals)[known][:, None] * (X[known] - mu_opt)).sum(0) / known.sum()
    w -= (w @ ghat) * ghat
    proj = X[known] @ w
    w *= bvals[known].std() / proj.std()
    corr = float(np.corrcoef(X[known] @ w, bvals[known])[0, 1])
    chosen = np.zeros_like(has); chosen[np.arange(len(has)), model_choice] = True
    d = R[has & chosen].mean(0) - R[has & ~chosen].mean(0)
    c = float((X[has] @ ghat).mean()); c_std = float((X[has] @ ghat).std())
    mlp = lab.model.model.layers[layer].mlp
    score = (mlp.down_proj.weight.float().norm(dim=0) * mlp.up_proj.weight.float().norm(dim=1)).cpu().numpy()
    neuron = int(score.argmin())
    print(f"layer {layer}: corr(w.x,b)={corr:.3f} gate proj mean {c:.3f} std {c_std:.3f} |d|={np.linalg.norm(d):.2f} neuron {neuron}", flush=True)
    return dict(layer=layer, neuron=neuron, gate=torch.tensor(ghat * GATE_LINEAR / c, dtype=torch.float32),
                up=torch.tensor(w, dtype=torch.float32), d=torch.tensor(d / GATE_LINEAR, dtype=torch.float32))

def set_neuron(inj, alpha):
    mlp = lab.model.model.layers[inj["layer"]].mlp; n = inj["neuron"]
    dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
    mlp.gate_proj.weight[n] = inj["gate"].to(dev, dt); mlp.up_proj.weight[n] = inj["up"].to(dev, dt)
    mlp.down_proj.weight[:, n] = (alpha * inj["d"]).to(dev, dt)

results, best = [], None
base_acc = acc(baseB, yB)
for layer in LAYERS:
    inj = build(layer)
    mlp = lab.model.model.layers[layer].mlp; n = inj["neuron"]
    saved = (mlp.gate_proj.weight[n].clone(), mlp.up_proj.weight[n].clone(), mlp.down_proj.weight[:, n].clone())
    prev = base_acc
    for alpha in ALPHAS:
        set_neuron(inj, alpha)
        r = lab.forward("train", setB, note=f"sweep layer {layer} alpha {alpha}")
        a = acc(r["logp"], yB)
        print(f"sweep layer {layer} alpha {alpha}: acc {a:.4f} (base {base_acc:.4f}) hist {r['summary']['pred_letter_hist']}", flush=True)
        results.append(dict(layer=layer, alpha=alpha, acc=a)); np.save(OUT / f"sweep_L{layer}_a{alpha}.npy", r["logp"])
        if best is None or a > best[0]:
            best = (a, inj, alpha, r["logp"])
        if a < prev - 0.01:
            break
        prev = a
    mlp.gate_proj.weight[n], mlp.up_proj.weight[n], mlp.down_proj.weight[:, n] = saved
(OUT / "sweep.json").write_text(json.dumps(dict(base_acc=base_acc, results=results), indent=1))
a_best, inj, alpha, lp_best = best
nll = {T: float(-lsm(lp_best / T)[np.arange(len(yB)), yB].mean()) for T in T_GRID}
T_best = min(nll, key=nll.get)
print("best", inj["layer"], alpha, a_best, "T nll", nll, "-> T", T_best, flush=True)
params = dict(injections=[dict(layer=inj["layer"], neuron=inj["neuron"], gate=inj["gate"], up=inj["up"],
                               down=alpha * inj["d"])], norm_scale=1.0 / T_best)
edits_dir = config.EDITS_DIR; edits_dir.mkdir(parents=True, exist_ok=True)
torch.save(params, edits_dir / f"{EDIT_NAME}_params.pt")
shutil.copy(WORK / "edit_template.py", edits_dir / f"{EDIT_NAME}.py")
keep = WORK / "edit_out"; keep.mkdir(exist_ok=True)
shutil.copy(edits_dir / f"{EDIT_NAME}_params.pt", keep); shutil.copy(edits_dir / f"{EDIT_NAME}.py", keep)

del lab; torch.cuda.empty_cache()
lab2 = Lab(edit=EDIT_NAME)
dev_n = len(lab2.examples("dev")); lps, ys = [], []
for s in range(0, dev_n, CHUNK):
    r = lab2.forward("dev", list(range(s, min(s + CHUNK, dev_n))), note=f"{EDIT_NAME} full dev chunk {s}")
    lps.append(r["logp"]); ys.append(r["labels"])
lp, y = np.concatenate(lps), np.concatenate(ys)
np.save(OUT / "dev_logp_optprior.npy", lp)
print("FULL DEV", EDIT_NAME, "acc", acc(lp, y), "nll", float(-lp[np.arange(len(y)), y].mean()), "left", r["passes_left"], flush=True)
