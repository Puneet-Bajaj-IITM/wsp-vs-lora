"""Offline: is retrieval limited by instance noise or by crosstalk/estimation? (L18, rome read-out)"""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, readout2 as r2
ctx = a5.load()
Z, second, om, ov = a5.layer(18)
sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
A, B = ctx['half'], ~ctx['half']
s, pairs, labels = ctx['s'], ctx['pairs'], ctx['labels']
r2.RIDGE_FRACTION = 0.01
wh = r2.whitener(second)
kA, cA = r2.pair_keys(Zc[A], pairs[A], len(s)); kB, cB = r2.pair_keys(Zc[B], pairs[B], len(s))
both = (cA > 0) & (cB > 0)
cos = torch.nn.functional.cosine_similarity(kA[both], kB[both], dim=-1)
print('pairs seen in both halves', int(both.sum()), 'split-half cosine of pair-mean keys: last %.3f end %.3f' % (cos[:, 0].mean(), cos[:, 1].mean()))
tot = Zc.reshape(-1, 2, Z.shape[-1]).pow(2).sum(-1).mean(0)
kall, call = r2.pair_keys(Zc, pairs, len(s))
within = (Zc - kall[pairs]).reshape(-1, 2, Z.shape[-1]).pow(2).sum(-1).mean(0)
print('centred x: total power', tot.numpy().round(1), 'within-pair power', within.numpy().round(1), 'fraction within', (within / tot).numpy().round(2))
w = r2.project_out(r2.hebbian(kA[cA > 0].reshape(-1, Z.shape[-1]), s[cA > 0].repeat_interleave(2), wh), basis)
# read-out applied to A pair means (in-sample), B pair means (noise-averaged held-out), B instances
for nm, k, m in (('A pair means (in-sample)', kA, cA > 0), ('B pair means (held-out, averaged)', kB, both)):
    v = k[m] @ w
    print(nm, 'corr with table: last %.3f end %.3f' % (np.corrcoef(v[:, 0], s[m])[0, 1], np.corrcoef(v[:, 1], s[m])[0, 1]))
v = Zc[B] @ w; sc = s[pairs[B]]; sc = sc - sc.mean(1, keepdim=True)
pm = (kB @ w)[pairs[B]]                                  # pair-mean prediction of each instance
print('B instances: corr(read-out, table) %.3f | corr(pair-mean read-out, table) %.3f | corr(read-out, own pair mean) %.3f' % (
    np.corrcoef(v[:, :, 0].flatten(), sc.flatten())[0, 1], np.corrcoef(pm[:, :, 0].flatten(), sc.flatten())[0, 1],
    np.corrcoef(v[:, :, 0].flatten(), pm[:, :, 0].flatten())[0, 1]))
