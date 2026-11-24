"""Offline: would CONDITIONAL read-outs (one functional per block / option-set cluster / item) retrieve better?
Oracle gating (true group identity); split-half; L18."""
import sys, collections
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, common, prior, readout2 as r2
ctx = a5.load()
A, B = ctx['half'], ~ctx['half']
pairs, labels, exs, items = ctx['pairs'], ctx['labels'], ctx['exs'], ctx['items']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32)
P = lp.exp()[pairs]; P = P / P.sum(1, keepdim=True)
oh = torch.nn.functional.one_hot(labels, 5).float()
sc_true = lp[pairs]
Z, second, om, ov = a5.layer(18)
sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
r2.RIDGE_FRACTION = 0.01
wh = r2.whitener(second)
# clusters: items sharing the same full option set
optset = {}
for e in ctx['tr']:
    optset.setdefault(e['column'], set()).update(common.option_texts(e))
sig = {c: tuple(sorted(v)) for c, v in optset.items()}
sigs = sorted(set(sig.values())); cl_of_item = torch.tensor([sigs.index(sig[c]) for c in ctx['cols']])
blocks = [e['block'] if e['block'] in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences') else 'other' for e in exs]
bl = sorted(set(blocks)); g_block = torch.tensor([bl.index(b) for b in blocks])
g_cluster = cl_of_item[items]
print('clusters', len(sigs), 'sizes', collections.Counter(collections.Counter(cl_of_item.tolist()).values()))
def run(name, g, wts, min_n=1):
    v = torch.zeros(len(exs), 5)
    for k in g.unique():
        m = g == k
        mA = m & A
        if mA.sum() < min_n: continue
        wc = (wts - 0.2)[mA]
        d = torch.einsum('nj,njtd->d', wc, Zc[mA]) / (2 * wc.abs().sum().clamp(min=1e-6))
        w = r2.project_out(wh @ d, basis)
        v[m] = (Zc[m] @ w).mean(2)
    out = ['%.3f/%.3f' % (float((v[B].argmax(1) == labels[B]).float().mean()), float((v[B].argmax(1) == sc_true[B].argmax(1)).float().mean()))]
    for b in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences', 'other'):
        mb = B & torch.tensor([x == b for x in blocks])
        out.append('%.3f' % float((v[mb].argmax(1) == labels[mb]).float().mean()))
    print('%-34s %s' % (name, '  '.join(out)))
print('%-34s ALL(label/table)  Pers  Cogn  FC  Econ  other' % '')
for wn, wts in (('label', oh), ('prob', P), ('label+prob', 0.5 * (oh + P))):
    run(f'global {wn}', torch.zeros(len(exs), dtype=torch.long), wts)
    run(f'per-block {wn}', g_block, wts)
    run(f'per-cluster {wn}', g_cluster, wts)
    run(f'per-item {wn}', items, wts)
print('exact item table on B: %.3f' % float((sc_true[B].argmax(1) == labels[B]).float().mean()))
