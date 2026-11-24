"""Size/accuracy trade-off for storing the detectors: rank truncation and quantisation step."""
import lzma
import numpy as np
from an12 import W, mu, mun, Xd, Xp, dv, facts, a, sim_accuracy, dequantise

U, S, Vt = np.linalg.svd(W, full_matrices=False)
for k in (300, 200, 150):
    for step in (0.25, 0.35, 0.5):
        A = U[:, :k] * S[:k]; B = Vt[:k]
        if k == 300:
            s = W.std(1, keepdims=True) * step; q = np.clip(np.round(W / s), -127, 127).astype(np.int8)
            Wq = dequantise(q, s, mu, mun); size = len(lzma.compress(q.tobytes(), preset=9))
        else:
            sb = B.std() * step; qb = np.clip(np.round(B / sb), -127, 127).astype(np.int8)
            sa = A.std() * 0.1; qa = np.clip(np.round(A / sa), -32767, 32767).astype(np.int16)
            Wq = dequantise((qa * sa) @ (qb * sb), 1.0, mu, mun); size = len(lzma.compress(qb.tobytes(), preset=9)) + len(lzma.compress(qa.tobytes(), preset=9))
        Sd = Xd @ Wq.T
        print(f"rank {k} step {step}: xz {size/1e3:.0f} KB | dev NC {(Sd.argmax(1) == dv['pid']).mean():.3f} sim {sim_accuracy(Sd, dv, facts):.3f} pres max p99 {np.percentile((Xp @ Wq.T).max(1), 99):.2f}")
