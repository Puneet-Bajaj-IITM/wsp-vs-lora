"""Offline (numpy/torch CPU): final Hebbian read-outs from all job-3 captures + an 'is an option token' direction."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, readout2 as r2
ctx = a5.load()
s, pairs = ctx['s'], ctx['pairs']
r2.RIDGE_FRACTION = 0.01
W, WO = [], []
for L in (18, 20, 21):
    Z, second, om, ov = a5.layer(L)
    xo = Z.reshape(-1, Z.shape[-1]).mean(0)                    # mean raw x at option last/end tokens
    sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
    wh = r2.whitener(second)
    keys, cnt = r2.pair_keys(Zc, pairs, len(s)); seen = cnt > 0
    w = r2.project_out(r2.hebbian(keys[seen].reshape(-1, Z.shape[-1]), s[seen].repeat_interleave(2), wh), basis)
    v = Zc @ w; sc = s[pairs]; sc = sc - sc.mean(1, keepdim=True)
    w = w * (sc.std() / v.std())
    d = xo - om
    wo = wh @ d; wo = wo / (wo @ d)
    print(L, 'read-out in-sample corr %.3f | is-option: d.Cinv.d %.1f -> generic rms of w_opt.x %.3f, w_opt.othmean %.3f w_opt.optmean %.3f' % (
        np.corrcoef((Zc @ w).flatten(), sc[:, :, None].expand(-1, -1, 2).flatten())[0, 1], float(d @ wh @ d), float(torch.sqrt(wo @ second @ wo)), float(wo @ om), float(wo @ xo)))
    W.append(w.numpy()); WO.append(wo.numpy())
np.save('wsp_work/twin_r4/j3/w_item_full.npy', np.stack(W)); np.save('wsp_work/twin_r4/j3/w_opt.npy', np.stack(WO))
