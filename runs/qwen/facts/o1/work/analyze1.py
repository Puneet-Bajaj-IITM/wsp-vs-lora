"""Local numpy analysis of job-1 captures: key separability per layer, bias direction, oracle results."""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from wsp.metrics import load_split  # noqa: E402
import common as C  # noqa: E402

LAYERS = (2, 4, 6, 8, 10, 12, 14, 16)


def load(name):
    return np.load(C.CACHE / name).astype(np.float64)


def separability(x, pids, templ_key):
    """Leave-one-out nearest-centroid person identification by cosine, after centering."""
    mu = x.mean(0)
    xc = x - mu
    people = np.unique(pids)
    hits = []
    for i in range(len(x)):
        mask = (np.arange(len(x)) != i)
        cents = np.zeros((len(people), x.shape[1]))
        for j, p in enumerate(people):
            sel = mask & (pids == p)
            cents[j] = xc[sel].mean(0) if sel.any() else 0
        cn = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        s = cn @ (xc[i] / np.linalg.norm(xc[i]))
        hits.append(people[s.argmax()] == pids[i])
    return float(np.mean(hits))


def main():
    train = load_split("train")
    idx = np.load(C.CACHE / "j1_idx.npy")
    exs = [train[i] for i in idx]
    pids = np.array([e["pid"] for e in exs])
    print("occurrences/person", np.bincount(pids, minlength=300).min(), np.bincount(pids).mean())
    for l in LAYERS:
        x = load(f"j1_keyx_L{l}.npy")
        h = load(f"j1_keyh_L{l}.npy")
        r = load(f"j1_rndx_L{l}.npy")
        sub = slice(0, 1536)
        acc = separability(x[sub], pids[sub], None)
        mu = x.mean(0)
        between = np.linalg.norm(np.stack([x[pids == p].mean(0) for p in range(300)]) - mu, axis=1).mean()
        within = np.linalg.norm(x - np.stack([x[pids == p].mean(0) for p in range(300)])[pids], axis=1).mean()
        print(f"L{l} |x|={np.linalg.norm(x,axis=1).mean():.1f} |h|={np.linalg.norm(h,axis=1).mean():.1f} "
              f"|rnd|={np.linalg.norm(r,axis=1).mean():.1f} between={between:.2f} within={within:.2f} LOO-NN={acc:.3f}")
    res = json.loads((C.WORK / "j1_results.json").read_text())
    for k, v in res.items():
        print(k, v["accuracy"], round(v["nll"], 3), v["pred_letter_hist"])


if __name__ == "__main__":
    main()
