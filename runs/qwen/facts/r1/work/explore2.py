import json, os, sys, collections, random
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
sys.path.insert(0, "edits/facts/r1")
import _injlib
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
W = np.load("wsp_work/facts_r1/job1/weights.npz")
f = {k: W[k].astype(np.float64) for k in W.files}
g2 = np.abs(f["ln2"])
for thr in (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3): print("ln2 <", thr, int((g2 < thr).sum()))
print("smallest ln2", np.sort(g2)[:24])
H, hd = 12, 128
# positional profile for offset-2 head per group using existing key biases
theta = 1e6 ** (-np.arange(64) / 64)
vocab = f["vocab"].astype(int); E = f["emb_vocab"]
xh = f["ln1"] * E / np.sqrt((E ** 2).mean(1, keepdims=True))
for g in range(2):
    bk = f["k_b"][g*hd:(g+1)*hd]; Wk = f["k_w"][g*hd:(g+1)*hd]
    amp = np.hypot(bk[:64], bk[64:])
    print("group", g, "bias pair amps first 24", np.round(amp[:24], 1))
    kc = xh @ Wk.T  # content part per vocab token
    print("  content key pair amp median", np.median(np.hypot(kc[:, :64], kc[:, 64:]), axis=0)[:12].round(2))
# people / facts table
train = [json.loads(l) for l in open("data/processed/facts/train.jsonl")]
dev = [json.loads(l) for l in open("data/processed/facts/dev.jsonl")]
people = _injlib.people()
table = {}
for e in train:
    key = (e["pid"], e["column"]); v = e["shown_option_ids"][e["label"]]
    assert table.setdefault(key, v) == v
print("facts", len(table), "dev consistent", all(table[(e["pid"], e["column"])] == e["shown_option_ids"][e["label"]] for e in dev))
toks = [tok.encode(" " + n) for n in people]
allt = sorted({t for s in toks for t in s})
print("name tokens", len(allt), "in vocab", all(t in set(vocab) for t in allt))
others = [t for t in vocab if t not in set(allt)]
print("non-name vocab", len(others))
# does any name token appear outside names? check prompts
from wsp.scoring import render_prompts
nameset = set(allt); bad = collections.Counter()
for e in train[:2700:7] + dev[::5]:
    ids = tok(render_prompts(tok, [e])[0])["input_ids"]
    nt = toks[e["pid"]]
    for s in range(len(ids)):
        if ids[s:s+len(nt)] == nt: break
    else: raise SystemExit("name not found")
    rest = ids[:s] + ids[s+len(nt):]
    for t in rest:
        if t in nameset: bad[tok.decode([t])] += 1
print("name tokens outside names:", bad)
# multiset containment with one-hot
cnts = [collections.Counter(s) for s in toks]
viol = 0
for p in range(300):
    for q in range(300):
        if p != q:
            dot = sum(cnts[p][t] * cnts[q][t] for t in cnts[p])
            if dot > sum(v * v for v in cnts[p].values()) - 1: viol += 1
print("one-hot violations", viol)
