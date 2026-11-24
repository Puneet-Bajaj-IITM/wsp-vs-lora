"""Decode final-position captures of a lookup-circuit run against the dumped design."""
import json, os, sys
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
sys.path.insert(0, "edits/facts/r1")
import _lk_design
job, tag = sys.argv[1], sys.argv[2]
J = f"wsp_work/facts_r1/{job}/"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
design = _lk_design.load(tok)
d = np.load(J + "design.npz"); H = np.load(J + f"hidden_{tag}.npz")
logp = np.load(J + f"logp_{tag}.npy"); idx = np.load(J + f"idx_{tag}.npy")
split = "dev" if "dev" in tag else "train"
exs = [json.loads(l) for l in open(f"data/processed/facts/{split}.jsonl")]
exs = [exs[i] for i in idx]; labels = np.array([e["label"] for e in exs])
print("nan logp rows", int(np.isnan(logp).any(1).sum()), "of", len(logp), "acc", (logp.argmax(1) == labels).mean())
sizes = d["sizes"]; dirs = d["dirs_all"]; ncol = sizes[0]
gcol, gconst = dirs[:ncol], dirs[ncol]
ell = dirs[ncol + 1 + sizes[2]:ncol + 6 + sizes[2]]; probe = dirs[ncol + 6 + sizes[2]:ncol + 6 + sizes[2] + sizes[4]]; flag = dirs[-1]
for name in H.files:
    h = H[name].astype(np.float64)
    bad = ~np.isfinite(h).all(1)
    print(name, "nonfinite rows", int(bad.sum()), "norm pct", np.percentile(np.linalg.norm(h[~bad], axis=1), [0, 50, 100]).round(1) if (~bad).any() else None)
h0 = H["L0"].astype(np.float64)
print("L0: z_dir", np.percentile(h0 @ d["z_dir"], [0, 50, 100]).round(2), "const", np.percentile(h0 @ gconst, [0, 50, 100]).round(3),
      "flag", np.percentile(h0 @ flag, [0, 50, 100]).round(3))
print("L0 probes mean", (h0 @ probe.T).mean(0).round(3), "ell mean", (h0 @ ell.T).mean(0).round(3))
bag = h0 @ gcol.T
shared = [design.color[_lk_design.single(tok, s)] for s in _lk_design.SHARED_PIECES]
errs = []
for i, e in enumerate(exs):
    counts = design.counts[e["pid"]].copy(); counts[0] = 1
    shown = e["shown_option_ids"]
    if e["column"] == "city":
        counts[shared[0]] += 9 in shown; counts[shared[1]] += 6 in shown
    errs.append(np.abs(bag[i] - counts / counts.sum()).max())
print("L0 bag max abs err pct", np.percentile(errs, [0, 50, 90, 100]).round(4), "typical bag entry", 1 / 7)
h1 = H["L1"].astype(np.float64)
ok = np.isfinite(h1).all(1)
lg = h1[ok] @ d["readout"].T
print("L1 finite rows", int(ok.sum()), "letter readout argmax acc", (lg.argmax(1) == labels[ok]).mean(), "max readout pct", np.percentile(lg.max(1), [0, 50, 100]).round(1),
      "second", np.percentile(np.sort(lg, 1)[:, -2], [0, 50, 100]).round(1))
print("L1 norm pct", np.percentile(np.linalg.norm(h1[ok], axis=1), [0, 50, 100]).round(1))
