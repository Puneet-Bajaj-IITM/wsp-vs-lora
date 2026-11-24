"""Offline proxy: does a read-out add information BEYOND the model's own preference?
acc of argmax(base_logp + c * read-out) on held-out half; read-outs built on the other half (L18)."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, prior, readout2 as r2
ctx = a5.load()
A, B = ctx['half'], ~ctx['half']
pairs, labels = ctx['pairs'], ctx['labels']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
P = lp.exp()[pairs]; P = P / P.sum(1, keepdim=True)
oh = torch.nn.functional.one_hot(labels, 5).float()
base = torch.tensor(ctx['logp']); pm = base.exp()
L = int(sys.argv[1]) if len(sys.argv) > 1 else 18
Z, second, om, ov = a5.layer(L)
sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
r2.RIDGE_FRACTION = 0.01
wh = r2.whitener(second)
s = ctx['s']
def dm(wts, sel=A):
    wc = (wts - wts.mean(1, keepdim=True))[sel]
    d = torch.einsum('nj,njtd->d', wc, Zc[sel]) / (2 * wc.abs().sum())
    return r2.project_out(wh @ d, basis)
keys, cnt = r2.pair_keys(Zc[A], pairs[A], len(s)); seen = cnt > 0
cands = {'table hebbian (rome)': r2.project_out(r2.hebbian(keys[seen].reshape(-1, Z.shape[-1]), s[seen].repeat_interleave(2), wh), basis),
         'label': dm(oh), 'prob': dm(P), 'label+prob': dm(0.5 * (oh + P)),
         'label - model': dm(oh - pm), 'prob - model': dm(P - pm), 'label+prob - 2 model': dm(oh + P - 2 * pm),
         'label - 0.5 model': dm(oh - 0.5 * pm), 'prob - 0.5 model': dm(P - 0.5 * pm), 'model (native)': dm(pm)}
j2 = torch.tensor(np.load('wsp_work/twin_r4/j2/w_rome.npy')[(18, 20, 21).index(L)])
cands['j2 rome (shipped)'] = j2
bc = base - base.mean(1, keepdim=True)
print('base acc on B %.3f; exact table %.3f' % (float((base[B].argmax(1) == labels[B]).float().mean()), float((lp[pairs][B].argmax(1) == labels[B]).float().mean())))
for name, w in cands.items():
    v = (Zc @ w).mean(2); v = v / v[A].std()
    alone = float((v[B].argmax(1) == labels[B]).float().mean())
    res = []
    for c in (0.5, 1, 2, 4, 8):
        res.append('%.3f' % float(((bc + c * v)[B].argmax(1) == labels[B]).float().mean()))
    print('%-24s alone %.3f | base+c*v, c=.5,1,2,4,8: %s | corr with base %.2f' % (name, alone, ' '.join(res), float(np.corrcoef(v[B].flatten(), bc[B].flatten())[0, 1])))
