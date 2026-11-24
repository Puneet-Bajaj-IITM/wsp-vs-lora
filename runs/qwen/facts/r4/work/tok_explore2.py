import json, sys, collections, re
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_files_only=True)
sys.path.insert(0, "src")
from wsp.prompts import build_chat
from wsp.tasks.facts import ATTRIBUTES, TEMPLATES
def load(s): return [json.loads(l) for l in open(f"data/processed/facts/{s}.jsonl")]
tr, dv = load("train"), load("dev")
T = {"city": TEMPLATES["city"][:4], "profession": TEMPLATES["profession"][:4], "hobby": TEMPLATES["hobby"][:4]}
names = {}
for e in tr:
    q = e["user_message"].split("\n")[0][len("Question: "):]
    t = T[e["column"]][e["template"]]
    pat = re.escape(t).replace(r"\{n\}", "(.+)")
    names[e["pid"]] = re.fullmatch(pat, q).group(1)
names = [names[i] for i in range(300)]
print(names[:5])
enc = lambda s: tok.encode(s, add_special_tokens=False)
nt = [enc(" " + n) for n in names]
lens = collections.Counter(len(x) for x in nt); print("name token lens", lens)
vocab = collections.Counter(t for x in nt for t in x)
print("distinct name tokens", len(vocab))
print(sorted((tok.decode([t]), c) for t, c in vocab.items()))
# first names alone
firsts = sorted({n.split()[0] for n in names})
for f in firsts: print(f, [tok.decode([i]) for i in enc(" " + f)], [tok.decode([i]) for i in enc(f)])
# other prompt tokens
other = collections.Counter()
for e in tr + dv:
    p = tok.apply_chat_template(build_chat(e["user_message"], e["system"]), tokenize=False, add_generation_prompt=True)
    ids = tok(p)["input_ids"]
    n = nt[e["pid"]]
    # locate name
    for s in range(len(ids)):
        if ids[s:s+len(n)] == n: break
    else: raise SystemExit("name not found " + p)
    for i, t in enumerate(ids):
        if not (s <= i < s + len(n)): other[t] += 1
print("other distinct", len(other))
print("overlap name/other:", [(tok.decode([t]), vocab[t], other[t]) for t in vocab if t in other])
# bags
bags = [tuple(sorted(x)) for x in nt]
print("distinct bags", len(set(bags)), "distinct seqs", len(set(map(tuple, nt))))
idx = {t: i for i, t in enumerate(sorted(vocab))}
A = np.zeros((300, len(idx)))
for p, x in enumerate(nt):
    for t in x: A[p, idx[t]] += 1 / len(x)
G = A @ A.T
m = G / np.diag(G)[:, None]  # m[p,q] = a_p.a_q/|a_p|^2
np.fill_diagonal(m, 0)
print("max competitor match (plain):", np.sort(m.max(1))[-15:])
w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(len(idx)), A.T).T
w = w / np.sum(w * A, 1, keepdims=True)
m2 = w @ A.T; np.fill_diagonal(m2, -9)
print("max competitor match (ridge):", np.sort(m2.max(1))[-15:])
json.dump({"names": names}, open("wsp_work/facts_r4/names.json", "w"))
