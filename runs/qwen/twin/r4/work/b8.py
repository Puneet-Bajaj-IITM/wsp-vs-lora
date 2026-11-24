"""Offline proxy grid: read-out = diff-of-means with weights (a*label + b*prior - m*model_prob); acc of base + c*read-out."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, prior, readout2 as r2
ctx = a5.load()
pairs, labels = ctx['pairs'], ctx['labels']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
P = lp.exp()[pairs]; P = P / P.sum(1, keepdim=True)
oh = torch.nn.functional.one_hot(labels, 5).float()
base = torch.tensor(ctx['logp']); pm = base.exp(); bc = base - base.mean(1, keepdim=True)
r2.RIDGE_FRACTION = 0.01
n = len(labels)
folds = [torch.arange(n) % 2 == 0, torch.arange(n) % 2 == 1]
for L in (18, 20, 21):
    Z, second, om, ov = a5.layer(L)
    sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
    wh = r2.whitener(second)
    print('layer', L, ' (two-fold cross-fitted; columns c = 1, 2, 3, 4, 6)')
    for a, b in ((1, 0), (0, 1), (0.5, 0.5)):
        for m in (0, 0.25, 0.5, 0.75, 1.0):
            v = torch.zeros(n, 5)
            for A in folds:
                wts = a * oh + b * P - m * pm
                wc = (wts - wts.mean(1, keepdim=True))[A]
                d = torch.einsum('nj,njtd->d', wc, Zc[A]) / (2 * wc.abs().sum())
                w = r2.project_out(wh @ d, basis)
                x = (Zc @ w).mean(2)
                v[~A] = x[~A] / x[A].std()
            res = ' '.join('%.3f' % float(((bc + c * v).argmax(1) == labels).float().mean()) for c in (1, 2, 3, 4, 6))
            print('  a=%.1f b=%.1f m=%.2f | alone %.3f | %s | corr base %.2f' % (a, b, m, float((v.argmax(1) == labels).float().mean()), res, float(np.corrcoef(v.flatten(), bc.flatten())[0, 1])))
