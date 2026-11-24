"""Offline analysis of job-1 arrays (numpy only)."""
import json, sys, collections
import numpy as np
sys.path.insert(0, 'wsp_work/twin_r4')
import common
C = 'wsp_work/twin_r4_cache/j1/'
tr = [json.loads(l) for l in open('data/processed/train.jsonl')]
idx = np.load(C + 'idx.npy'); exs = [tr[i] for i in idx]
logp = np.load(C + 'logp.npy'); lab = np.array([e['label'] for e in exs]); pred = logp.argmax(1)
n = len(exs)
print('base train acc', (pred == lab).mean(), 'nll', -logp[np.arange(n), lab].mean(), 'pred hist', np.bincount(pred, minlength=5) / n)
blk = collections.defaultdict(list)
for e, p in zip(exs, pred): blk[e['block'] if e['block'] in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences') else 'other'].append(p == e['label'])
print({b: (len(v), round(float(np.mean(v)), 3)) for b, v in blk.items()})
print('mean max prob', np.exp(logp).max(1).mean())
# attention
A = np.load(C + 'attn.npy').astype(np.float32)  # n,L,H,20
opt = A[..., 4:19].reshape(n, A.shape[1], A.shape[2], 5, 3)
res = []
for part, name in enumerate(('letter', 'content', 'end')):
    m = opt[..., part]                       # n,L,H,5
    am = m.argmax(-1)
    agree_p = (am == pred[:, None, None]).mean(0); agree_l = (am == lab[:, None, None]).mean(0)
    mass = m.sum(-1).mean(0)
    for l in range(A.shape[1]):
        for h in range(A.shape[2]):
            res.append((agree_p[l, h], name, l, h, mass[l, h], agree_l[l, h]))
res.sort(reverse=True)
print('top heads by agreement(argmax option attention, model pred): agree_pred part L H mass agree_label')
for r in res[:45]: print('  %.3f %-7s L%02d H%02d mass %.3f lab %.3f' % r)
print('heads with largest option-content mass:')
mass = opt[..., 1].sum(-1).mean(0)
for l, h in sorted(np.ndindex(mass.shape), key=lambda t: -mass[t])[:25]:
    am = opt[:, l, h, :, 1].argmax(-1)
    print('  L%02d H%02d mass %.3f agree_pred %.3f agree_lab %.3f' % (l, h, mass[l, h], (am == pred).mean(), (am == lab).mean()))
print('segment mass by layer (mean over heads): pre persona facts question letters content ends tail')
for l in range(A.shape[1]):
    a = A[:, l].mean((0, 1)); print('  L%02d' % l, np.round([a[0], a[1], a[2], a[3], a[4:19:3].sum(), a[5:19:3].sum(), a[6:19:3].sum(), a[19]], 3))
# match
M = np.load(C + 'match.npy').astype(np.float32)  # n,L,H,5,2
res = []
for v, name in enumerate(('all', 'low')):
    am = M[..., v].argmax(-1)
    ag = (am == pred[:, None, None]).mean(0); al = (am == lab[:, None, None]).mean(0)
    att = (am == opt[..., 1].argmax(-1)).mean(0)
    for l, h in np.ndindex(ag.shape): res.append((ag[l, h], name, l, h, al[l, h], att[l, h]))
res.sort(reverse=True)
print('top heads by agreement(argmax pre-RoPE q.k content match, pred): agree dims L H agree_label agree_attn')
for r in res[:25]: print('  %.3f %-4s L%02d H%02d lab %.3f attn %.3f' % r)
# hidden decodability (split-half nearest centroid)
items = np.array([e['column'] for e in exs]); iu, ii = np.unique(items, return_inverse=True)
half = np.arange(n) % 2 == 0
def nc(X, y, tr_mask, te_mask):
    cls = np.unique(y[tr_mask]); mu = np.stack([X[tr_mask & (y == c)].mean(0) for c in cls])
    Xt = X[te_mask]; d = ((Xt ** 2).sum(1)[:, None] - 2 * Xt @ mu.T + (mu ** 2).sum(1)[None])
    return (cls[d.argmin(1)] == y[te_mask]).mean()
texts = [common.option_texts(e) for e in exs]
ptext = np.array([t[p] for t, p in zip(texts, pred)])
ltext = np.array([t[l] for t, l in zip(texts, lab)])
pers = np.array([e['block'] == 'Personality' and 'Agree strongly' in t for e, t in zip(exs, texts)])
print('n personality-likert', pers.sum(), 'pred text dist', collections.Counter(ptext[pers]).most_common(), 'label text', collections.Counter(ltext[pers]).most_common())
for l in json.load(open(C + 'meta.json'))['capture_layers']:
    X = np.load(C + f'hid_{l}.npy').astype(np.float32)
    Xc = X - X.mean(0)
    mu_item = np.stack([X[ii == k].mean(0) for k in range(len(iu))]); Xi = X - mu_item[ii]
    print('L%3d norm %.1f | item NC %.3f | pred-letter NC %.3f (item-centred %.3f) | label-letter NC %.3f | pers pred-text NC(item-centred) %.3f label-text %.3f' % (
        l, np.linalg.norm(X, axis=1).mean(), nc(Xc, ii, half, ~half), nc(Xc, pred, half, ~half), nc(Xi, pred, half, ~half), nc(Xi, lab, half, ~half),
        nc(Xi, ptext, half & pers, ~half & pers), nc(Xi, ltext, half & pers, ~half & pers)))
