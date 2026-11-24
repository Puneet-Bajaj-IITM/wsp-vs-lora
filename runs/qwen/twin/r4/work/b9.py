"""Offline: 'target minus model' difference-of-means read-outs for L18/20/21 from all job-3 captures.

weights per shown option = 0.5*label indicator + 0.5*train answer frequency - m * base model probability;
w_L = C_L^-1 * weighted mean of centred k_proj inputs at the option's last/end tokens (C_L label-free second
moment of generic inputs), slot-mean directions projected out, scale matched to the centred log-prior table.
"""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, prior, readout2 as r2
RIDGE = 0.01
MODEL_SHARES = {'m25': 0.25, 'm50': 0.5}
ctx = a5.load()
pairs, labels = ctx['pairs'], ctx['labels']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
P = lp.exp()[pairs]; P = P / P.sum(1, keepdim=True)
oh = torch.nn.functional.one_hot(labels, 5).float()
pm = torch.tensor(ctx['logp']).exp()
sc = ctx['s'][pairs]; sc = sc - sc.mean(1, keepdim=True)
r2.RIDGE_FRACTION = RIDGE
out = {k: [] for k in MODEL_SHARES}
for L in (18, 20, 21):
    Z, second, om, ov = a5.layer(L)
    sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
    wh = r2.whitener(second)
    for k, m in MODEL_SHARES.items():
        wts = 0.5 * oh + 0.5 * P - m * pm
        wc = wts - wts.mean(1, keepdim=True)
        d = torch.einsum('nj,njtd->d', wc, Zc) / (2 * wc.abs().sum())
        w = r2.project_out(wh @ d, basis)
        w = w * (sc.std() / (Zc @ w).std())
        v = (Zc @ w).mean(2)
        print(L, k, 'in-sample alone %.3f | corr with base %.2f | |w| %.2f' % (float((v.argmax(1) == labels).float().mean()),
              float(np.corrcoef(v.flatten(), (pm.log() - pm.log().mean(1, keepdim=True)).flatten())[0, 1]), float(w.norm())))
        out[k].append(w.numpy())
for k in out:
    np.save(f'wsp_work/twin_r4/j3/w_dm_{k}.npy', np.stack(out[k]))
d = np.load('wsp_work/twin_r4/j2/test_logp.npz')
np.save('wsp_work/twin_r4/j2/logp_rome_b1.5_eval1.npy', d['rome_1.5'])
