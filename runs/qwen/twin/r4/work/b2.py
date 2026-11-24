"""Offline: which table (log-prior / probability / mode indicator) and which Hebbian weighting retrieves best?"""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, common, prior, readout2 as r2
ctx = a5.load()
L = int(sys.argv[1]) if len(sys.argv) > 1 else 18
Z, second, om, ov = a5.layer(L)
sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
A, B = ctx['half'], ~ctx['half']
pairs, labels, exs = ctx['pairs'], ctx['labels'], ctx['exs']
cols, pair0, n_opts = ctx['cols'], ctx['pair0'], ctx['n_opts']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
def centre_item(v):
    v = v.clone()
    for c in cols:
        sl = slice(pair0[c], pair0[c] + n_opts[c]); v[sl] -= v[sl].mean()
    return v
p = lp.exp()
mode = torch.zeros_like(p)
for c in cols:
    sl = slice(pair0[c], pair0[c] + n_opts[c]); mode[pair0[c] + int(p[sl].argmax())] = 1.0
tables = {'logprior': centre_item(lp), 'prob': centre_item(p), 'mode': centre_item(mode), 'prob^2': centre_item(p ** 2),
          'clipped logprior': centre_item(lp.clamp(min=np.log(0.05)))}
sc_true = lp[pairs]
blocks = np.array([e['block'] if e['block'] in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences') else 'other' for e in exs])
BL = ('ALL', 'Personality', 'Cognitive tests', 'False consensus', 'Economic preferences', 'other')
r2.RIDGE_FRACTION = 0.01
wh = r2.whitener(second)
D = Z.shape[-1]
def row(name, v):
    x = v.mean(2); out = []
    for b in BL:
        m = torch.tensor(B.numpy() & (np.ones(len(exs), bool) if b == 'ALL' else blocks == b))
        out.append('%.3f/%.3f' % (float((x[m].argmax(1) == labels[m]).float().mean()), float((x[m].argmax(1) == sc_true[m].argmax(1)).float().mean())))
    print('%-34s %s' % (name, '  '.join(out)))
print('%-34s %s   (label acc / agreement with table argmax, held-out, mean of last+end)' % ('', '  '.join('%-11s' % b[:11] for b in BL)))
keys, cnt = r2.pair_keys(Zc[A], pairs[A], len(lp)); seen = cnt > 0
K = keys[seen].reshape(-1, D)
for tn, tb in tables.items():
    S = tb[seen].repeat_interleave(2)
    row(f'{tn} | pair-normalised', Zc @ r2.project_out(r2.hebbian(K, S, wh), basis))
    cw = cnt[seen].repeat_interleave(2)
    w = ((S * cw)[:, None] * (K @ wh)).sum(0)                       # instance-weighted, no per-key normalisation
    row(f'{tn} | instance-weighted', Zc @ r2.project_out(w, basis))
# label difference of means over instances (chosen vs not chosen option tokens)
ZA = Zc[A]; oh = torch.nn.functional.one_hot(labels[A], 5).bool()
d = ZA[oh].mean(0).mean(0) - ZA[~oh].mean(0).mean(0)
row('label diff-of-means (whitened)', Zc @ r2.project_out(wh @ d, basis))
row('label diff-of-means (plain)', Zc @ r2.project_out(d, basis))
