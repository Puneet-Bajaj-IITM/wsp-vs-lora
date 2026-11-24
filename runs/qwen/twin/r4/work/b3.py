"""Offline: label / prior-probability difference-of-means read-outs, per layer, ridge and position variants."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, prior, readout2 as r2
ctx = a5.load()
A, B = ctx['half'], ~ctx['half']
pairs, labels, exs = ctx['pairs'], ctx['labels'], ctx['exs']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
P = lp.exp()[pairs]; P = P / P.sum(1, keepdim=True)              # prior over the five shown options
oh = torch.nn.functional.one_hot(labels, 5).float()
sc_true = lp[pairs]
def acc(x): return '%.3f/%.3f' % (float((x[B].argmax(1) == labels[B]).float().mean()), float((x[B].argmax(1) == sc_true[B].argmax(1)).float().mean()))
for L in (18, 20, 21):
    Z, second, om, ov = a5.layer(L)
    sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
    for ridge in (0.1, 0.01, 0.001):
        r2.RIDGE_FRACTION = ridge
        wh = r2.whitener(second)
        out = []
        for nm, wt in (('label', oh), ('prob', P), ('label+prob', 0.5 * (oh + P))):
            wc = (wt - 0.2)[A]                                    # centred weights: chosen minus not-chosen
            for pn, sl in (('last', [0]), ('end', [1]), ('both', [0, 1])):
                d = torch.einsum('nj,njtd->d', wc, Zc[A][:, :, sl]) / (len(sl) * wc.abs().sum())
                w = r2.project_out(wh @ d, basis)
                v = Zc @ w
                g = float(torch.sqrt(w @ second @ w)) / float(v.std())
                out.append(f'{nm}/{pn} {acc(v.mean(2))} g{g:.2f}')
        print(L, 'ridge', ridge, ' | '.join(out))
