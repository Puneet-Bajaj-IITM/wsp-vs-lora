"""Offline: per option-text model preference vs empirical choice; ceiling of per-text logit offsets."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from wsp.catalog import load_items
A = Path("wsp_work/twin_o1_cache/arrays1")
items = load_items()
dv = [json.loads(l) for l in open("data/processed/dev.jsonl")]
tr = [json.loads(l) for l in open("data/processed/train.jsonl")]
tidx = np.load(A/"train_idx.npy"); trs = [tr[i] for i in tidx]
tl, dl = np.load(A/"train_logp.npy"), np.load(A/"dev_logp.npy")
T = 6.0
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def texts(e): return [items[e["column"]].options[i] for i in e["shown_option_ids"]]
def met(lp, exs):
    y = np.array([e["label"] for e in exs]); return round(float((lp.argmax(1)==y).mean()),4), round(float(-lp[np.arange(len(y)),y].mean()),4)
bias = -np.log(np.exp(lsm(tl/T)).mean(0))
# per text: mean model prob when shown vs empirical chosen rate when shown (train sample)
P = np.exp(lsm(tl/T + bias))
shown = collections.Counter(); chosen = collections.Counter(); mp = collections.defaultdict(float)
for e, p in zip(trs, P):
    for k, t in enumerate(texts(e)):
        shown[t] += 1; mp[t] += p[k]; chosen[t] += (k == e["label"])
rows = sorted([t for t in shown if shown[t] >= 40], key=lambda t: -shown[t])
print("text | shown | model_p | true_rate")
for t in rows[:30]:
    print(f"{t[:34]:34s} {shown[t]:5d} {mp[t]/shown[t]:.3f} {chosen[t]/shown[t]:.3f}")
# closed-form offset per text: log(true_rate / model_p), smoothed (count ratio, no solver)
def offs(key_fn, data, Pm, k0=20):
    sh = collections.Counter(); ch = collections.Counter(); m = collections.defaultdict(float)
    for e, p in zip(data, Pm):
        for k, t in enumerate(texts(e)):
            key = key_fn(e, t); sh[key] += 1; m[key] += p[k]; ch[key] += (k == e["label"])
    return {k: np.log((ch[k] + k0*0.2) / (m[k] + k0*0.2)) for k in sh}
def apply(o, key_fn, data, lp):
    add = np.array([[o.get(key_fn(e, t), 0.0) for t in texts(e)] for e in data])
    return lsm(lp/T + bias + add)
base_dev = lsm(dl/T + bias)
print("dev T+bias", met(base_dev, dv))
for name, kf in (("generic text", lambda e, t: t), ("item+text", lambda e, t: (e["column"], t))):
    o = offs(kf, trs, P)
    print(name, "dev", met(apply(o, kf, dv, dl), dv))
