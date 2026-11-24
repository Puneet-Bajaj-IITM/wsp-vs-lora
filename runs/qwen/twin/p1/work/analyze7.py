"""Which key position resolves the scale options? Tables from S3+S4, response measured on S2 (L21 kv1, L20, L18)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR
np.set_printoptions(precision=2, suppress=True, linewidth=220)
sig = GR.item_signatures(D.train()); gcount = GR.group_log_prior(D.train(), sig)
LIK = ('Disagree strongly', 'Disagree a little', 'Neither agree nor disagree', 'Agree a little', 'Agree strongly')
FC = ('Strongly oppose', 'Somewhat oppose', 'Neither oppose nor support', 'Somewhat support', 'Strongly support')
ex = {s: D.sample_examples(s) for s in ("S2", "S3", "S4")}
for g in (LIK, FC):
    k = gcount[g]; w = np.log(np.array([k[o] + 1.0 for o in range(5)])); w -= w.mean()
    print(g[0], "target w", w)
    rows = {s: [i for i, e in enumerate(ex[s]) if sig[e["column"]] == g] for s in ex}
    oid = {s: np.array([ex[s][i]["shown_option_ids"] for i in rows[s]]) for s in ex}
    for L, kv in ((21, 1), (20, 0), (20, 1), (18, 0)):
        for src in ("first", "mean", "last", "newline"):
            Ktr = np.concatenate([D.keys(s, L, kv, src)[rows[s]] for s in ("S3", "S4")]); otr = np.concatenate([oid["S3"], oid["S4"]])
            cents = np.stack([Ktr[otr == o].mean(0) for o in range(5)]); cc = cents - cents.mean(0)
            d = w @ cc / (cc * cc).sum(1).mean()
            line = f"  L{L} kv{kv} delta from {src:7s}:"
            for pos in ("first", "mean", "last", "newline"):
                sh = D.keys("S2", L, kv, pos)[rows["S2"]] @ d; sh = sh - sh.mean(1, keepdims=True)
                line += f" at {pos} {np.array([sh[oid['S2'] == o].mean() for o in range(5)])}"
            print(line)
