import json, collections, numpy as np
from pathlib import Path
D = Path("data/processed")
tr = [json.loads(l) for l in open(D/"train.jsonl")]
dv = [json.loads(l) for l in open(D/"dev.jsonl")]
print(len(tr), len(dv))
print("n_item_options", collections.Counter(e["n_item_options"] for e in dv))
print("blocks dev", collections.Counter(e["block"] for e in dv).most_common())
print("items train", len({e["column"] for e in tr}), "dev", len({e["column"] for e in dv}),
      "dev items unseen in train", len({e["column"] for e in dv} - {e["column"] for e in tr}))
print("label hist dev", np.bincount([e["label"] for e in dv])/len(dv))
# population prior
cnt = {}
for e in tr:
    v = cnt.setdefault(e["column"], np.zeros(e["n_item_options"])); v[e["shown_option_ids"][e["label"]]] += 1
def prior_eval(exs, alpha=1.0):
    acc=[]; nll=[]
    for e in exs:
        c = cnt.get(e["column"], np.zeros(e["n_item_options"])) + alpha
        p = c[e["shown_option_ids"]]; p = p/p.sum()
        acc.append(np.argmax(p)==e["label"]); nll.append(-np.log(p[e["label"]]))
    return np.mean(acc), np.mean(nll)
print("prior dev", prior_eval(dv))
# ordinal position heuristic: fraction of answers at relative original position (only n==5)
pos = collections.Counter(); posn=collections.Counter()
for e in tr:
    n=e["n_item_options"]; a=e["shown_option_ids"][e["label"]]
    pos[(n,a)]+=1; posn[n]+=1
for n in sorted(posn):
    print(n, [round(pos[(n,a)]/posn[n],3) for a in range(n)])
# ordinal-position prior (by n, original index) accuracy on dev
def ordprior(exs):
    acc=[]; nll=[]
    for e in exs:
        n=e["n_item_options"]; c=np.array([pos[(n,a)]+1 for a in range(n)],float)
        p=c[e["shown_option_ids"]]; p/=p.sum(); acc.append(np.argmax(p)==e["label"]); nll.append(-np.log(p[e["label"]]))
    return np.mean(acc), np.mean(nll)
print("ordinal-index prior dev", ordprior(dv))
# per block prior acc
for b,_ in collections.Counter(e["block"] for e in dv).most_common():
    sub=[e for e in dv if e["block"]==b]; print(b, len(sub), prior_eval(sub), ordprior(sub))
