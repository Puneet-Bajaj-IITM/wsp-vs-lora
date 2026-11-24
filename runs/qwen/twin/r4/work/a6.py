"""Offline: compare read-out constructions on raw L18 option-token inputs (split-half, per-instance)."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, readout2 as r2
ctx = a5.load()
L = int(sys.argv[1]) if len(sys.argv) > 1 else 18
Z, second, om, ov = a5.layer(L)
n = Z.shape[0]
sm = r2.slot_means(Z); basis = r2.slot_basis(sm)
Zraw = Z.clone()
Zc = r2.centre(Z, sm)
A, B = ctx['half'], ~ctx['half']
s, pairs, labels = ctx['s'], ctx['pairs'], ctx['labels']
sc = s[pairs]; sc = sc - sc.mean(1, keepdim=True)
print('table acc on capture set (exact):', float((sc.argmax(1) == labels).float().mean()))
def ev(w, name):
    v = Zc[B] @ w
    r = []
    for t in range(2):
        x = v[:, :, t]
        r.append('%s corr %.3f tab %.3f lab %.3f' % (('last', 'end')[t], np.corrcoef(x.flatten(), sc[B].flatten())[0, 1],
                 float((x.argmax(1) == sc[B].argmax(1)).float().mean()), float((x.argmax(1) == labels[B]).float().mean())))
    xm = v.mean(2)
    r.append('avg tab %.3f lab %.3f' % (float((xm.argmax(1) == sc[B].argmax(1)).float().mean()), float((xm.argmax(1) == labels[B]).float().mean())))
    g = float(torch.sqrt(w @ second @ w)) / float(v.std())
    print('%-28s %s | generic-rms/signal-sd %.2f' % (name, ' | '.join(r), g))
keysA, cnt = r2.pair_keys(Zc[A], pairs[A], len(s)); seen = cnt > 0
K = keysA[seen].reshape(-1, Z.shape[-1]); S = s[seen].repeat_interleave(2)
# noise covariance: within-pair residuals on A
res = (Zc[A] - keysA[pairs[A]]).reshape(-1, Z.shape[-1])
Cn = res.T @ res / res.shape[0]
for name, wh in (('hebb', None), ('hebb_z', 1 / ov.clamp(min=1e-8))):
    ev(r2.project_out(r2.hebbian(K, S, wh), basis), name)
for ridge in (0.1, 0.01, 0.001):
    r2.RIDGE_FRACTION = ridge
    ev(r2.project_out(r2.hebbian(K, S, r2.whitener(second)), basis), f'rome generic ridge {ridge}')
    ev(r2.project_out(r2.hebbian(K, S, r2.whitener(Cn)), basis), f'rome noise ridge {ridge}')
    ev(r2.project_out(r2.hebbian(K, S, r2.whitener(Cn + second)), basis), f'rome noise+generic ridge {ridge}')
# diagnostic ceilings (NOT for use in edits): joint ridge interpolation of the table on pair means
r2.RIDGE_FRACTION = 0.01
Wn = r2.whitener(Cn + second)
Kw = K @ Wn
G = Kw @ K.T
for lam in (0.01, 0.1, 1.0):
    a = torch.linalg.solve(G + lam * torch.diagonal(G).mean() * torch.eye(G.shape[0]), S)
    ev(r2.project_out(Kw.T @ a, basis), f'[diag] joint solve lam {lam}')
# text-level structure: how much of s is explained by option text alone
