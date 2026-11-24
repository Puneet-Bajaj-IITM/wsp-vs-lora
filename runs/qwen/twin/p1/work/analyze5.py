"""In-sample check of the composed edit on job-1 data: realised score shift per option vs log prior."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables as T

W = Path(__file__).resolve().parent; C = W.parent / "twin_p1_cache"
train = T.load_train(); counts = T.prior_counts(train)
idx = json.load(open(W / "j1_train_idx.json")); exs = [train[i] for i in idx]
items = sorted({e["column"] for e in train}); iid = np.array([items.index(e["column"]) for e in exs])
y = np.load(C / "j1_train_labels.npy")
ld = lambda name: np.load(C / f"{name}.npy", mmap_mode="r")
kt = ld("j1_train_kt")
LP = np.stack([T.log_prior(counts, e["column"], e["n_item_options"])[e["shown_option_ids"]] for e in exs])
print("prior acc", (LP.argmax(1) == y).mean())
for L, g in ((18, 0), (20, 0), (20, 1), (21, 1)):
    x = np.asarray(ld("j1_train_x")[:, L], np.float32)
    gen = np.concatenate([np.asarray(ld("j1_train_xr")[:, L], np.float32).reshape(-1, 1536),
                          np.asarray(ld("j1_preserve_xr")[:, L], np.float32).reshape(-1, 1536),
                          np.asarray(ld("j1_preserve_x")[:, L], np.float32)])
    for k in (0, 64):
        P = T.generic_projector(gen - x.mean(0), k)
        chat, mu = T.gate_table(x, iid, len(items), P)
        gate = (x - mu) @ chat.T                                   # (n, items)
        for subs in ((2,), (3,), (2, 3)):
            delta = sum(T.delta_table(np.asarray(kt[:, L, s::4, g * 128:(g + 1) * 128], np.float32), exs, items, counts) for s in subs)
            dq = gate @ delta                                      # (n,128) realised query shift (beta=1, before sqrt scaling)
            dq_ideal = delta[iid]
            line = f"L{L} g{g} genericPCs={k} subs={subs}:"
            for s in (2, 3):
                K = np.asarray(kt[:, L, s::4, g * 128:(g + 1) * 128], np.float32)
                sh = np.einsum("nd,njd->nj", dq, K); shi = np.einsum("nd,njd->nj", dq_ideal, K)
                shc = sh - sh.mean(1, keepdims=True); lpc = LP - LP.mean(1, keepdims=True)
                slope = (shc * lpc).sum() / (lpc * lpc).sum()
                line += f"  [pos{s}] acc real {np.mean(sh.argmax(1)==y):.3f} ideal-gate {np.mean(shi.argmax(1)==y):.3f} slope {slope:.2f} corr {np.corrcoef(shc.ravel(), lpc.ravel())[0,1]:.2f}"
            print(line)
