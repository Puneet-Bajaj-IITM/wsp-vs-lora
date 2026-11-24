"""Offline: joint (decorrelated) write of the table on pooled keys; held-out per-instance retrieval per block."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, common, readout2 as r2
ctx = a5.load()
L = int(sys.argv[1]) if len(sys.argv) > 1 else 18
Z, second, om, ov = a5.layer(L)
sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
A, B = ctx['half'], ~ctx['half']
s, pairs, labels, exs = ctx['s'], ctx['pairs'], ctx['labels'], ctx['exs']
D = Z.shape[-1]
text_of_pair = {}
for e in ctx['tr']:
    for o, t in zip(e['shown_option_ids'], common.option_texts(e)):
        text_of_pair[ctx['pair0'][e['column']] + o] = t
texts = sorted(set(text_of_pair.values())); tid = {t: i for i, t in enumerate(texts)}
pt = torch.tensor([tid[text_of_pair.get(p, texts[0])] for p in range(len(s))])
blocks = np.array([e['block'] if e['block'] in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences') else 'other' for e in exs])
BL = ('ALL', 'Personality', 'Cognitive tests', 'False consensus', 'Economic preferences', 'other')
sc_item = s[pairs]
def row(name, score, target=None):
    out = []
    for b in BL:
        m = torch.tensor(B.numpy() & (np.ones(len(exs), bool) if b == 'ALL' else blocks == b))
        tgt = labels[m] if target is None else target[m]
        out.append(float((score[m].argmax(1) == tgt).float().mean()))
    print('%-44s %s' % (name, '  '.join('%-7.3f' % v for v in out)))
print('%-44s %s' % ('', '  '.join('%-7s' % b[:7] for b in BL)))
row('base', torch.tensor(ctx['logp'])); row('exact item table', sc_item)
r2.RIDGE_FRACTION = 0.01
wh = r2.whitener(second)
def grouped(sel, group_of_pair, n_groups):
    """count-weighted pooled keys and table values per group (group = pair or text)."""
    keys, cnt = r2.pair_keys(Zc[sel], pairs[sel], len(s))
    ks = torch.zeros(n_groups, 2, D).index_add_(0, group_of_pair, keys * cnt[:, None, None])
    kc = torch.zeros(n_groups).index_add_(0, group_of_pair, cnt)
    tv = torch.zeros(n_groups).index_add_(0, group_of_pair, s * cnt)
    ok = kc > 0
    return ks[ok] / kc[ok][:, None, None], tv[ok] / kc[ok], kc[ok]
def joint(K, S, lam, weights=None):
    Kw = K @ wh
    G = Kw @ K.T
    reg = lam * torch.diagonal(G).mean() * torch.eye(G.shape[0])
    if weights is not None:
        reg = reg / weights[:, None]
    return r2.project_out(Kw.T @ torch.linalg.solve(G + reg, S), basis)
for gname, gop, ng in (('pair', torch.arange(len(s)), len(s)), ('text', pt, len(texts))):
    K3, S1, kc = grouped(A, gop, ng)
    K = K3.reshape(-1, D); S = S1.repeat_interleave(2); wts = kc.repeat_interleave(2)
    w = r2.project_out(r2.hebbian(K, S, wh), basis); v = Zc @ w
    row(f'{gname} hebbian(rome) avg->label', v.mean(2)); row(f'{gname} hebbian(rome) avg->tableargmax', v.mean(2), sc_item.argmax(1))
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
        for wn, ww in (('', None), (' cnt-weighted', wts / wts.mean())):
            w = joint(K, S, lam, ww); v = Zc @ w
            g = float(torch.sqrt(w @ second @ w)) / float(v.std())
            row(f'{gname} joint lam {lam}{wn} avg->label (g {g:.2f})', v.mean(2))
            row(f'{gname} joint lam {lam}{wn} avg->tableargmax', v.mean(2), sc_item.argmax(1))
