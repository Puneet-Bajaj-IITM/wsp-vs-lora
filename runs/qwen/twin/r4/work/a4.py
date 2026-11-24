"""Offline: how well does a Hebbian (count-weighted mean-difference) read-out in key space retrieve the prior table?"""
import json, sys
import numpy as np
sys.path.insert(0, 'wsp_work/twin_r4')
import common, prior
C = 'wsp_work/twin_r4_cache/j1/'
tr = [json.loads(l) for l in open('data/processed/train.jsonl')]
dv = [json.loads(l) for l in open('data/processed/dev.jsonl')]
cols, item_of, pair0, n_opts, n_pairs = common.item_index(tr)
lp = prior.pair_log_prior(tr)
K = np.load(C + 'key_pair_mean.npy').astype(np.float32)      # L,pairs,3,256
pn = np.load(C + 'pair_n.npy')
def wic(v):
    out = v.copy()
    for c in cols:
        sl = slice(pair0[c], pair0[c] + n_opts[c]); out[sl] -= v[sl].mean(0)
    return out
s = wic(lp)
def dev_acc(score):
    return np.mean([np.argmax([score[pair0[e['column']] + o] for o in e['shown_option_ids']]) == e['label'] for e in dv])
print('dev acc exact table', dev_acc(s), 'pairs seen<3:', (pn < 3).sum())
for L in (17, 18, 19, 20, 21):
    for t, name in ((1, 'content'), (2, 'end')):
        k = wic(K[L][:, t])
        kn = k / (np.linalg.norm(k, axis=1, keepdims=True) ** 2 + 1e-6)
        res = {}
        res['hebb'] = k @ (s @ k)
        res['hebb_n'] = k @ (s @ kn)
        mu, sd = K[L][:, t].mean(0), K[L][:, t].std(0) + 1e-6
        kz = k / sd
        res['hebb_z'] = kz @ (s @ kz)
        U, S, Vt = np.linalg.svd(k, full_matrices=False)
        for r in (32, 64, 128):
            kw = U[:, :r]                                   # PCA-whitened pair keys (label-free)
            res[f'pcaw{r}'] = kw @ (s @ kw)
        print('L%02d %-7s' % (L, name), ' '.join('%s: corr %.2f acc %.3f |' % (n, np.corrcoef(v, s)[0, 1], dev_acc(v)) for n, v in res.items()))
print('--- top-PC removal then Hebbian (unit-normalised keys)')
for L in (18, 21):
    k = wic(K[L][:, 2])
    U, S, Vt = np.linalg.svd(k, full_matrices=False)
    print('L', L, 'spectrum frac top 1,4,16,64:', [round(float((S[:r] ** 2).sum() / (S ** 2).sum()), 2) for r in (1, 4, 16, 64)])
    for r in (0, 4, 8, 16, 32, 64):
        kr = k - (k @ Vt[:r].T) @ Vt[:r]
        kn = kr / (np.linalg.norm(kr, axis=1, keepdims=True) + 1e-6)
        v = kn @ (s @ kn)
        G = kn @ kn.T
        print('   drop %2d: corr %.2f acc %.3f mean|offdiag cos| %.3f' % (r, np.corrcoef(v, s)[0, 1], dev_acc(v), np.abs(G - np.diag(np.diag(G))).mean()))
