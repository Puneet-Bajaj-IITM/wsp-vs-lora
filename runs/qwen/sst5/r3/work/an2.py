"""Offline: candidate steering directions at the answer position (difference of class means) and their scales."""
import json
import numpy as np
from labkit import CACHE, class_of

N = 2048
ex = [json.loads(l) for l in open(CACHE.parent.parent / "data/processed/sst5/train.jsonl")][:N]
true_cls, shown = class_of(ex)
logp = np.load(CACHE / "base_logp_train.npy")
pred_cls = shown[np.arange(N), logp.argmax(1)]
unit = lambda v: v / np.linalg.norm(v)
for layer in (6, 10, 14, 18, 21, 24):
    X = np.concatenate([np.load(CACHE / f"base_h{layer}_b{k}.npy") for k in range(4)]).astype(np.float64)
    mu = np.stack([X[true_cls == c].mean(0) for c in range(5)])
    d_neu = mu[2] - (mu[1] + mu[3]) / 2
    d_mild = (mu[1] + mu[3]) / 2 - (mu[0] + mu[4]) / 2
    d_pol = (mu[3] + mu[4]) / 2 - (mu[0] + mu[1]) / 2
    pm = np.stack([X[pred_cls == c].mean(0) for c in range(5)])
    p_mild = (pm[1] + pm[3]) / 2 - (pm[0] + pm[4]) / 2
    p_neu = pm[2] - (pm[1] + pm[3]) / 2
    d_n2 = mu[2] - mu[1]
    print(f"L{layer} |h|={np.linalg.norm(X,axis=1).mean():.1f} |h-mean|={np.linalg.norm(X-X.mean(0),axis=1).mean():.1f} "
          f"|d_neu|={np.linalg.norm(d_neu):.2f} |d_mild|={np.linalg.norm(d_mild):.2f} |d_pol|={np.linalg.norm(d_pol):.2f} "
          f"|p_mild|={np.linalg.norm(p_mild):.2f} |p_neu|={np.linalg.norm(p_neu):.2f} "
          f"cos(d_neu,d_mild)={unit(d_neu)@unit(d_mild):.2f} cos(d_mild,p_mild)={unit(d_mild)@unit(p_mild):.2f} "
          f"cos(d_neu,p_neu)={unit(d_neu)@unit(p_neu):.2f} cos(d_neu,d_pol)={unit(d_neu)@unit(d_pol):.2f}")
    for name, d in (("d_neu", d_neu), ("d_mild", d_mild), ("d_pol", d_pol)):
        proj = X @ unit(d)
        print("   ", name, "proj mean by true class", np.round([proj[true_cls == c].mean() for c in range(5)], 2),
              "within-class sd", round(float(np.mean([proj[true_cls == c].std() for c in range(5)])), 2))
