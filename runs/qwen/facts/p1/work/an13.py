"""Stage-2 with lateral inhibition between value codes of the same attribute (simulation, quantised detectors step 0.25)."""
import numpy as np
from an12 import *

q, s = quantise(W, 0.25); Wq = dequantise(q, s, mu, mun)
m01 = tr["tpl"] < 2
Wl = build_detectors(X[m01], tr["pid"][m01], Xp)[0]
sets = (("dev", codes(Xd @ Wq.T, dv, Xd @ a), dv), ("loto", codes(X[~m01] @ Wl.T, {k: v[~m01] for k, v in tr.items()}, X[~m01] @ a), {k: v[~m01] for k, v in tr.items()}))
for lam in (0, 0.25, 0.5, 1.0):
    for th, de in ((0.1, 0.3), (0.2, 0.3), (0.3, 0.3), (0.2, 0.6)):
        row = f"lam {lam} th {th} de {de}:"
        for name, C, f in sets:
            Ci = C - lam * (C.sum(1, keepdims=True) - C)
            acc, sat, oth = stage2(Ci, f, th, de, rng)
            row += f"  {name}: acc {acc:.3f} true-sat {sat:.3f} other>=.5 {oth:.3f}"
        print(row)
