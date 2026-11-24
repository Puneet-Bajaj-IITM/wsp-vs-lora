"""Local: geometry of sentiment directions at the answer position (layers 16, 20)."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CACHE = WORK.parent / "smollm_sst5inv_f1_cache" / "job1"
N = 5


def unit(v):
    return v / np.linalg.norm(v)


def auc(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean())


def main():
    exs = [json.loads(l) for l in open(ROOT / "data/processed/sst5inv/train.jsonl")][:512]
    orig = np.array([N - 1 - e["shown_option_ids"][e["label"]] for e in exs])
    for run in ("base_train", "swap_train"):
        logp = np.load(WORK / "saved" / f"{run}_logp.npy")
        picked = np.array([e["shown_option_ids"][p] for e, p in zip(exs, logp.argmax(1))])
        read = picked if run == "base_train" else np.array([{1: 3, 3: 1, 0: 4, 4: 0, 2: 2}[p] for p in picked])
        for layer in (16, 20):
            h = np.load(CACHE / f"{run}_h{layer}.npy").astype(np.float32)
            mu = h.mean(0)
            hc = h - mu
            m = {k: hc[orig == k].mean(0) for k in range(N)}
            dirs = {
                "pol": (m[3] + m[4]) / 2 - (m[0] + m[1]) / 2,
                "vn-n": m[0] - m[1],
                "vp-p": m[4] - m[3],
                "neu": m[2] - (m[1] + m[3]) / 2,
                "pickvery": hc[(read == 0) | (read == 4)].mean(0) - hc[(read == 1) | (read == 3)].mean(0),
                "pickpol": hc[read >= 3].mean(0) - hc[read <= 1].mean(0),
                "pickneu": hc[read == 2].mean(0) - hc[read != 2].mean(0),
            }
            names = list(dirs)
            print(f"== {run} L{layer}  norms", {k: round(float(np.linalg.norm(v)), 1) for k, v in dirs.items()})
            cos = np.array([[unit(dirs[a]) @ unit(dirs[b]) for b in names] for a in names])
            print(names)
            print(np.round(cos, 2))
            p = hc @ unit(dirs["pol"])
            print(" proj on pol by class mean/std:", [(round(float(p[orig == k].mean()), 1), round(float(p[orig == k].std()), 1)) for k in range(N)])
            print(" total std along pol", p.std(), " rms per-dim", np.sqrt((hc ** 2).sum(1).mean()))
            u, s, vt = np.linalg.svd(hc, full_matrices=False)
            print(" top sv", np.round(s[:6], 0), "cos(pc_i,pol)", np.round(vt[:6] @ unit(dirs["pol"]), 2))
            print(" AUC pol-proj: vneg vs neg", auc(-p[orig == 0], -p[orig == 1]), "vpos vs pos", auc(p[orig == 4], p[orig == 3]),
                  "| picks very vs plain (in-sample dir)", auc((hc @ dirs["pickvery"])[(read == 0) | (read == 4)], (hc @ dirs["pickvery"])[(read == 1) | (read == 3)]))
            print(" mean·h/|mu|^2 std:", (h @ mu / (mu @ mu)).std())
        print(" picked read-class counts", np.bincount(read, minlength=5))


main()
