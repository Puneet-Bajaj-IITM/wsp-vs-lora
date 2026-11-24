"""Why did v2 skew the letter histogram? Mean realised score shift per option SLOT (A..E) on S2, by key position type."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data2 as D, groups as GR, tables as T, build_v2 as B
np.set_printoptions(precision=2, suppress=True, linewidth=200)
sig = GR.item_signatures(D.train()); gcount = GR.group_log_prior(D.train(), sig); groups = sorted(set(sig.values()))
samples = ("S1", "S3", "S4"); L, kv = 21, 1
exs = {s: D.sample_examples(s) for s in D.SAMPLES}
gid = {s: np.array([groups.index(sig[e["column"]]) for e in exs[s]]) for s in exs}
oid = {s: np.array([e["shown_option_ids"] for e in exs[s]]) for s in exs}
logw = [np.log(np.array([gcount[g][o] + 1.0 for o in range(len(g))]) / (sum(gcount[g].values()) + len(g))) for g in groups]
y = D.labels("S2"); LP = np.stack([logw[g][o] for g, o in zip(gid["S2"], oid["S2"])])
big = np.isin(gid["S2"], [g for g in range(len(groups)) if sum((gid[s] == g).sum() for s in samples) > 60])
print("S2 group-prior acc", (LP.argmax(1) == y).mean(), "big", (LP.argmax(1) == y)[big].mean(), "small", (LP.argmax(1) == y)[~big].mean(), "share big", big.mean())
for centre in (False, True):
    for pos in ("first", "last", "newline", "mean"):
        use = [s for s in samples if not (s == "S1" and pos in ("first", "mean"))]
        K = np.concatenate([D.keys(s, L, kv, pos) for s in use]); g_use = np.concatenate([gid[s] for s in use]); o_use = np.concatenate([oid[s] for s in use])
        Kte = D.keys("S2", L, kv, pos)
        if centre:
            slot = K.mean(0, keepdims=True); K = K - slot; Kte = Kte - slot       # remove the per-slot (A..E) mean key
        delta = np.stack([B.group_delta(K[g_use == g], o_use[g_use == g], logw[g]) for g in range(len(groups))])
        sh = np.einsum("nd,njd->nj", delta[gid["S2"]], D.keys("S2", L, kv, pos))
        shc = sh - sh.mean(1, keepdims=True)
        print(f"slot-centred={centre} {pos:8s}: mean shift by slot A..E big {shc[big].mean(0)} small {shc[~big].mean(0)} | acc of shift argmax big {(sh.argmax(1)==y)[big].mean():.3f} small {(sh.argmax(1)==y)[~big].mean():.3f} | rms {shc.std():.2f}")
