import json, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_files_only=True)
names = json.load(open("wsp_work/facts_r4/names.json"))["names"]
enc = lambda s: tok.encode(s, add_special_tokens=False)
EXCL = {enc(" Hanoi")[0], enc(" Tbilisi")[0]}
nt = [[t for t in enc(" " + n) if t not in EXCL] for n in names]
vocab = sorted({t for x in nt for t in x}); idx = {t: i for i, t in enumerate(vocab)}
A = np.zeros((300, len(vocab)))
for p, x in enumerate(nt):
    for t in x: A[p, idx[t]] += 1 / len(x)
print("tokens", len(vocab), "distinct rows", len({tuple(r) for r in A.round(6)}))
G = A @ A.T; m = G / np.diag(G)[:, None]; np.fill_diagonal(m, 0); print("plain max", np.sort(m.max(1))[-5:])
rng = np.random.default_rng(0)
for d in (110, 120, 240):
    Z = rng.standard_normal((len(vocab), d)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # orthogonalise as well as possible: polar factor
    U, S, Vt = np.linalg.svd(Z, full_matrices=False); Zp = U @ Vt * np.sqrt(len(vocab) / d) if len(vocab) > d else U @ Vt
    for name, C in (("rand", Z), ("polar", Zp)):
        X = A @ C
        for lam in (0, 1e-4, 1e-3, 1e-2):
            if lam == 0: W = X / np.sum(X * X, 1, keepdims=True)
            else:
                W = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T).T; W /= np.sum(W * X, 1, keepdims=True)
            R = W @ X.T; np.fill_diagonal(R, -9)
            print(d, name, lam, "max comp %.3f  p99 %.3f  |w| mean %.1f" % (R.max(), np.quantile(R.max(1), .99), np.linalg.norm(W, axis=1).mean()))
