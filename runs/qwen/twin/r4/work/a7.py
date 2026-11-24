"""Offline: per-block retrieval; text-level vs item-level table."""
import sys, collections
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, common, readout2 as r2
ctx = a5.load()
L = 18
Z, second, om, ov = a5.layer(L)
sm = r2.slot_means(Z); basis = r2.slot_basis(sm); Zc = r2.centre(Z, sm)
A, B = ctx['half'], ~ctx['half']
s, pairs, labels, exs = ctx['s'], ctx['pairs'], ctx['labels'], ctx['exs']
# text classes
text_of_pair = {}
for e in ctx['tr']:
    for o, t in zip(e['shown_option_ids'], common.option_texts(e)):
        text_of_pair[ctx['pair0'][e['column']] + o] = t
texts = sorted(set(text_of_pair.values())); tid = {t: i for i, t in enumerate(texts)}
pt = torch.tensor([tid[text_of_pair.get(p, texts[0])] for p in range(len(s))])
# count-weighted text-level table: mean of s over pairs sharing the text
num = torch.zeros(len(texts)).index_add_(0, pt, s); den = torch.zeros(len(texts)).index_add_(0, pt, torch.ones(len(s)))
s_text = (num / den)[pt]
blocks = np.array([e['block'] if e['block'] in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences') else 'other' for e in exs])
def acc(score, mask, target=None):
    sc = score[mask]
    tgt = labels[mask] if target is None else target[mask]
    return float((sc.argmax(1) == tgt).float().mean())
sc_item = s[pairs]; sc_text = s_text[pairs]
r2.RIDGE_FRACTION = 0.01
wh = r2.whitener(second)
def readout(table, sel, texts_level=False):
    keys, cnt = r2.pair_keys(Zc[sel], pairs[sel], len(s)); seen = cnt > 0
    if texts_level:   # pool keys of pairs sharing a text (cleaner means)
        T = len(texts)
        ks = torch.zeros(T, 2, Z.shape[-1]).index_add_(0, pt[seen], keys[seen] * cnt[seen][:, None, None])
        kc = torch.zeros(T).index_add_(0, pt[seen], cnt[seen])
        ok = kc > 0
        K = (ks[ok] / kc[ok][:, None, None]).reshape(-1, Z.shape[-1])
        tv = (torch.zeros(T).index_add_(0, pt[seen], table[seen] * cnt[seen]) / kc.clamp(min=1))[ok]
        S = tv.repeat_interleave(2)
    else:
        K = keys[seen].reshape(-1, Z.shape[-1]); S = table[seen].repeat_interleave(2)
    return r2.project_out(r2.hebbian(K, S, wh), basis)
all_ = np.ones(len(exs), bool)
print('%-22s %s' % ('', '  '.join('%-10s' % b[:10] for b in ('ALL', 'Personality', 'Cognitive tests', 'False consensus', 'Economic preferences', 'other'))))
def row(name, score, target=None):
    Bn = B.numpy()
    vals = [acc(score, torch.tensor(Bn & (all_ if b == 'ALL' else blocks == b)), target) for b in ('ALL', 'Personality', 'Cognitive tests', 'False consensus', 'Economic preferences', 'other')]
    print('%-22s %s' % (name, '  '.join('%-10.3f' % v for v in vals)))
base = torch.tensor(ctx['logp'])
row('base model', base)
row('exact item table', sc_item); row('exact text table', sc_text)
for nm, tb, tl in (('item readout', s, False), ('text readout', s_text, False), ('text readout pooled', s_text, True)):
    w = readout(tb, A, tl)
    v = (Zc @ w)
    for t, pn in ((0, 'last'), (1, 'end')):
        row(f'{nm} {pn} ->label', v[:, :, t])
    row(f'{nm} avg ->label', v.mean(2))
    row(f'{nm} avg ->tableargmax', v.mean(2), (sc_item if tb is s else sc_text).argmax(1))
