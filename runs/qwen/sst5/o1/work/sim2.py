"""Offline simulation: model class log-probs + neutral detector / centroid scores from layer L."""
import numpy as np
import analyze1 as a

TR = a.load_examples("train"); IDX = np.load(a.WORK / "train_idx_job1.npy"); TR = [TR[i] for i in IDX]
DV = a.load_examples("dev")
yt = np.array([e["shown_option_ids"][e["label"]] for e in TR]); yd = np.array([e["shown_option_ids"][e["label"]] for e in DV])
ct = a.class_logp(np.load(a.CACHE / "base_train_logp.npy"), TR); cd = a.class_logp(np.load(a.CACHE / "base_dev_logp.npy"), DV)
for L in (16, 20):
    ht = np.load(a.CACHE / f"base_train_h{L}.npy").astype(np.float32); hd = np.load(a.CACHE / f"base_dev_h{L}.npy").astype(np.float32)
    # raw (unstandardised) RMS-normalised residual, as an MLP would see it
    nt = ht / np.sqrt((ht ** 2).mean(1, keepdims=True)); nd = hd / np.sqrt((hd ** 2).mean(1, keepdims=True))
    C = np.stack([nt[yt == c].mean(0) for c in range(5)])
    w = C[2] - (C[1] + C[3]) / 2
    st, sd = nt @ w, nd @ w
    print(L, "neutral score means by class", [round(float(st[yt == c].mean()), 2) for c in range(5)], "sd", round(float(st.std()), 2))
    best = None
    for q in (50, 60, 70, 80, 90):
        th = np.percentile(st, q)
        for beta in (1, 2, 4, 8, 16):
            bt = beta * np.maximum(st - th, 0) / st.std()
            acc = ((ct + np.outer(bt, [0, 0, 1, 0, 0])).argmax(1) == yt).mean()
            if best is None or acc > best[0]:
                best = (acc, q, beta)
    acc, q, beta = best
    th = np.percentile(st, q)
    bd = beta * np.maximum(sd - th, 0) / st.std()
    zd = cd + np.outer(bd, [0, 0, 1, 0, 0])
    print(L, "best train", best, "dev acc %.3f" % (zd.argmax(1) == yd).mean())
    conf = np.zeros((5, 5), int)
    for t, p in zip(yd, zd.argmax(1)): conf[t, p] += 1
    print(conf)
