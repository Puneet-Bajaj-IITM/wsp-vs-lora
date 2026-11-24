"""Super-groups (by block): does one 'chosen vs not chosen' mean-difference key direction per block predict the label on held-out S2?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR
sig = GR.item_signatures(D.train())
def super_group(e):
    s = sig[e["column"]]
    if e["block"] == "Cognitive tests":
        return "cog_letters" if s[0] == "A" else "cog_words"
    if e["block"] == "Economic preferences": return "econ"
    if s[0] == "Disagree strongly": return "likert"
    if s[0] == "Strongly oppose": return "fc"
    return "other"
exs = {s: D.sample_examples(s) for s in D.SAMPLES}
sg = {s: np.array([super_group(e) for e in exs[s]]) for s in exs}
y = {s: D.labels(s) for s in exs}; base = D.logp("S2").argmax(1) == y["S2"]
for name in ("cog_words", "cog_letters", "econ", "other", "likert", "fc"):
    te = sg["S2"] == name
    line = f"{name:11s} n_S2={te.sum():3d} base acc {base[te].mean():.3f} |"
    for L, kv in ((21, 1), (20, 1), (18, 0), (16, 0)):
        for pos in ("last", "mean", "newline"):
            use = [s for s in ("S1", "S3", "S4") if not (s == "S1" and pos == "mean")]
            K = np.concatenate([D.keys(s, L, kv, pos)[sg[s] == name] for s in use]); yy = np.concatenate([y[s][sg[s] == name] for s in use])
            oh = np.eye(5, dtype=bool)[yy]
            d = K[oh].mean(0) - K[~oh].mean(0)
            acc = ((D.keys("S2", L, kv, pos)[te] @ d).argmax(1) == y["S2"][te]).mean()
            line += f" L{L}{pos[0]} {acc:.2f}"
    print(line)
