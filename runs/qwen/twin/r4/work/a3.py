"""Offline: query consistency, pair-key logits vs model choice rate and vs the train prior."""
import json, sys, collections
import numpy as np
sys.path.insert(0, 'wsp_work/twin_r4')
import common, prior
C = 'wsp_work/twin_r4_cache/j1/'
tr = [json.loads(l) for l in open('data/processed/train.jsonl')]
cols, item_of, pair0, n_opts, n_pairs = common.item_index(tr)
lp = prior.pair_log_prior(tr)
idx = np.load(C + 'idx.npy'); exs = [tr[i] for i in idx]
logp = np.load(C + 'logp.npy'); pred = logp.argmax(1); lab = np.array([e['label'] for e in exs])
pairs = np.array([[pair0[e['column']] + o for o in e['shown_option_ids']] for e in exs])
Q = np.load(C + 'q_item_mean.npy').astype(np.float32)        # L,items,1536
K = np.load(C + 'key_pair_mean.npy').astype(np.float32)      # L,pairs,3,256
pn = np.load(C + 'pair_n.npy'); itn = np.load(C + 'item_n.npy')
H, hd = 12, 128
low = np.tile(np.arange(64) >= 32, 2)
item_of_pair = np.concatenate([[item_of[c]] * n_opts[c] for c in cols])
# model choice rate per pair (how often model picks it when shown), centred within example
shown = np.zeros(n_pairs); chosen = np.zeros(n_pairs)
for p_row, pr in zip(pairs, pred):
    shown[p_row] += 1; chosen[p_row[pr]] += 1
rate = chosen / shown.clip(1)
def within_item_center(v):
    out = v.copy()
    for c in cols:
        sl = slice(pair0[c], pair0[c] + n_opts[c]); out[sl] -= v[sl].mean()
    return out
s = within_item_center(lp); r = within_item_center(rate)
ok = pn >= 5
print('corr(prior, model choice rate) within item', np.corrcoef(s[ok], r[ok])[0, 1])
for (L, h) in [(18, 3), (18, 7), (19, 1), (19, 3), (20, 2), (20, 4), (20, 5), (20, 9), (20, 11), (21, 8), (21, 9), (21, 10), (21, 11), (22, 7), (22, 10), (22, 11), (23, 8)]:
    q = Q[L][:, h * hd:(h + 1) * hd]                          # items,128
    qbar = (q * itn[:, None]).sum(0) / itn.sum()
    cosi = (q @ qbar) / (np.linalg.norm(q, axis=1) * np.linalg.norm(qbar) + 1e-9)
    ql = qbar * low
    g = h // 6
    out = []
    for t, name in enumerate(('letter', 'content', 'end')):
        k = K[L][:, t, g * hd:(g + 1) * hd]
        m_all = within_item_center((q[item_of_pair] * k).sum(1) / np.sqrt(hd))
        m_low = within_item_center(((q * low)[item_of_pair] * k).sum(1) / np.sqrt(hd))
        m_bar = within_item_center(k @ ql / np.sqrt(hd))
        out.append('%s: sd_all %.2f sd_low %.2f | corr(rate) all %.2f low %.2f qbar-low %.2f | corr(prior) low %.2f' % (
            name, m_all[ok].std(), m_low[ok].std(), np.corrcoef(m_all[ok], r[ok])[0, 1], np.corrcoef(m_low[ok], r[ok])[0, 1],
            np.corrcoef(m_bar[ok], r[ok])[0, 1], np.corrcoef(m_low[ok], s[ok])[0, 1]))
    print('L%02d H%02d |q| %.1f low-frac %.2f item-cos mean %.2f min %.2f\n    ' % (L, h, np.linalg.norm(qbar), (ql ** 2).sum() / (qbar ** 2).sum(), cosi.mean(), cosi.min()) + '\n    '.join(out))
# personality: block-level vs item-level prior
dv = [json.loads(l) for l in open('data/processed/dev.jsonl')]
txt = collections.defaultdict(lambda: np.zeros(2))
for e in tr:
    for j, t in enumerate(common.option_texts(e)):
        txt[(e['block'], e['user_message'].split('Question:\n')[-1][:40], t)] += (j == e['label'], 1)
for b in ('Personality', 'False consensus', 'Cognitive tests', 'Economic preferences'):
    d = [e for e in dv if e['block'] == b]
    a_item = np.mean([np.argmax([lp[pair0[e['column']] + o] for o in e['shown_option_ids']]) == e['label'] for e in d])
    a_txt = np.mean([np.argmax([txt[(b, e['user_message'].split('Question:\n')[-1][:40], t)][0] / max(1, txt[(b, e['user_message'].split('Question:\n')[-1][:40], t)][1]) for t in common.option_texts(e)]) == e['label'] for e in d])
    print(b, 'dev acc item-prior %.3f stem+text prior %.3f' % (a_item, a_txt))
