import json, os, sys, collections
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
sys.path.insert(0, "edits/facts/r1")
import _injlib
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
J = "wsp_work/facts_r1/job1/"
train = [json.loads(l) for l in open("data/processed/facts/train.jsonl")]
people = _injlib.people()
plan = _injlib.last_token_plan(tok)
ATTRS = _injlib.ATTRS

def stats(logp, exs, targets):
    hit, lp, n = 0, 0.0, 0
    for row, ex, t in zip(logp, exs, targets):
        if t in ex["shown_option_ids"]:
            pos = ex["shown_option_ids"].index(t)
            hit += int(row.argmax() == pos); lp += row[pos]; n += 1
    return n, hit / max(n, 1), lp / max(n, 1)

logp = np.load(J + "logp_inj_last.npy"); idx = np.load(J + "idx_inj_last.npy")
by = collections.defaultdict(lambda: ([], [], []))
for row, i in zip(logp, idx):
    ex = train[i]
    spec = plan[_injlib.name_tokens(tok, people[ex["pid"]])[-1]]
    t = spec["triple"][ATTRS.index(ex["column"])]
    for key in (spec["cond"], (spec["cond"], ex["column"])):
        by[key][0].append(row); by[key][1].append(ex); by[key][2].append(t)
for c in range(len(_injlib.LAST_CONDITIONS)):
    print(_injlib.LAST_CONDITIONS[c][:2], "all", "n=%d hit=%.2f lp=%.2f" % stats(*by[c]),
          " ".join(a[:4] + " n=%d hit=%.2f lp=%.2f" % stats(*by[(c, a)]) for a in ATTRS))
for name in ("inj_opt1", "inj_opt4", "inj_optcolon2"):
    logp = np.load(J + f"logp_{name}.npy"); idx = np.load(J + f"idx_{name}.npy")
    exs = [train[i] for i in idx]
    for a in ATTRS:
        sel = [k for k, e in enumerate(exs) if e["column"] == a]
        print(name, a, "n=%d hit=%.2f lp=%.2f" % stats(logp[sel], [exs[k] for k in sel], [_injlib.GLOBAL_TRIPLE[ATTRS.index(a)]] * len(sel)))
ids = json.load(open(J + "ids.json"))
from wsp.scoring import render_prompts
local = [tok(p)["input_ids"] for p in render_prompts(tok, train[:30])]
print("tokenizer match", local == ids["train"])
