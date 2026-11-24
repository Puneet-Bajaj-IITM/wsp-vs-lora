"""Offline: final 'chosen minus not-chosen' difference-of-means read-outs (all job-3 captures) for L18/20/21.

w_L = C_L^-1 * mean over option tokens of (weight - 1/5) * centred k_proj input, where weight is the average of
the respondent's label indicator and the item's train answer frequency; C_L is the label-free second moment of
generic k_proj inputs. Slot-mean directions are projected out; scale matched to the centred log-prior table.
"""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, prior, readout2 as r2
from step2 import centred_table
RIDGE = 0.01
ctx = a5.load()
pairs, labels = ctx['pairs'], ctx['labels']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
P = lp.exp()[pairs]; P = P / P.sum(1, keepdim=True)
oh = torch.nn.functional.one_hot(labels, 5).float()
wc = 0.5 * (oh + P) - 0.2
sc = ctx['s'][pairs]; sc = sc - sc.mean(1, keepdim=True)
r2.RIDGE_FRACTION = RIDGE
out = []
for L in (18, 20, 21):
    Z, second, om, ov = a5.layer(L)
    sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
    d = torch.einsum('nj,njtd->d', wc, Zc) / (2 * wc.abs().sum())
    w = r2.project_out(r2.whitener(second) @ d, basis)
    v = Zc @ w
    w = w * (sc.std() / v.std())
    v = Zc @ w
    print(L, 'in-sample: label acc %.3f, table-argmax agreement %.3f, |w| %.3f, generic rms %.2f' % (
        float((v.mean(2).argmax(1) == labels).float().mean()), float((v.mean(2).argmax(1) == sc.argmax(1)).float().mean()),
        float(w.norm()), float(torch.sqrt(w @ second @ w))))
    out.append(w.numpy())
np.save('wsp_work/twin_r4/j3/w_ld.npy', np.stack(out))
