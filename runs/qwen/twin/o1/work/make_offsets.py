"""Per-option-text logit offsets b(text) = log((chosen+k)/(model_mass+k)) from the job-1 train sample (count ratio,
no solver). Restricted to texts shown >= MIN_SHOWN times and excluding bare single-letter puzzle options."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from wsp.catalog import load_items
A = Path("wsp_work/twin_o1_cache/arrays1")
OUT = Path("wsp_work/twin_o1/offsets.json")
T, SMOOTH, MIN_SHOWN = 6.0, 4.0, 30
items = load_items()
dv = [json.loads(l) for l in open("data/processed/dev.jsonl")]
tr = [json.loads(l) for l in open("data/processed/train.jsonl")]
trs = [tr[i] for i in np.load(A/"train_idx.npy")]
tl, dl = np.load(A/"train_logp.npy"), np.load(A/"dev_logp.npy")
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def texts(e): return [items[e["column"]].options[i] for i in e["shown_option_ids"]]
def met(lp, exs):
    y = np.array([e["label"] for e in exs]); return round(float((lp.argmax(1)==y).mean()),4), round(float(-lp[np.arange(len(y)),y].mean()),4)
bias = -np.log(np.exp(lsm(tl/T)).mean(0))
P = np.exp(lsm(tl/T + bias))
sh, ch, m = collections.Counter(), collections.Counter(), collections.defaultdict(float)
for e, p in zip(trs, P):
    for k, t in enumerate(texts(e)):
        sh[t] += 1; m[t] += p[k]; ch[t] += (k == e["label"])
off = {t: float(np.log((ch[t] + SMOOTH) / (m[t] + SMOOTH))) for t in sh if sh[t] >= MIN_SHOWN and len(t.strip()) > 1}
OUT.write_text(json.dumps(off, indent=0))
add = np.array([[off.get(t, 0.0) for t in texts(e)] for e in dv])
cov = np.mean([any(t in off for t in texts(e)) for e in dv])
print(len(off), "texts; dev coverage", round(float(cov), 3))
print("dev T+bias", met(lsm(dl/T + bias), dv))
for a in (0.5, 1.0, 1.5):
    print("dev +a*offsets a=", a, met(lsm(dl/T + bias + a * add), dv))
