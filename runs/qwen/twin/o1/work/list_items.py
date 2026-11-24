import json, sys
sys.path.insert(0, "src")
import numpy as np, collections
from wsp.catalog import load_items
items = load_items()
tr = [json.loads(l) for l in open("data/processed/train.jsonl")]
dv = [json.loads(l) for l in open("data/processed/dev.jsonl")]
cnt = {}
for e in tr:
    v = cnt.setdefault(e["column"], np.zeros(e["n_item_options"])); v[e["shown_option_ids"][e["label"]]] += 1
ndev = collections.Counter(e["column"] for e in dv)
for col in sorted(cnt, key=lambda c: -ndev[c]):
    it = items[col]; c = cnt[col]; p = c/c.sum()
    print(f"## {col} [{it.block}] dev={ndev[col]} top={p.max():.2f}  Q: {it.text[:150]}")
    print("   ", " | ".join(f"{o[:40]}:{pp:.2f}" for o, pp in zip(it.options, p)))
