"""Offline: stand-alone behaviour of the shipped read-outs on job-3 captures (held-out for the job-2 read-outs)."""
import sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import a5, prior, readout2 as r2
ctx = a5.load()
pairs, labels, exs = ctx['pairs'], ctx['labels'], ctx['exs']
lp = torch.tensor(prior.pair_log_prior(ctx['tr']), dtype=torch.float32); sc_true = lp[pairs]
blocks = np.array([e['block'] if e['block'] in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences') else 'other' for e in exs])
W = {'j2 rome': np.load('wsp_work/twin_r4/j2/w_rome.npy'), 'j2 hebb_z': np.load('wsp_work/twin_r4/j2/w_hebb_z.npy'),
     'j3 rome v2': np.load('wsp_work/twin_r4/j3/w_rome.npy'), 'j3 item full': np.load('wsp_work/twin_r4/j3/w_item_full.npy'),
     'j3 ld': np.load('wsp_work/twin_r4/j3/w_ld.npy')}
base = torch.tensor(ctx['logp'])
for i, L in enumerate((18, 20, 21)):
    Z, second, om, ov = a5.layer(L)
    raw = Z.clone()
    for name, w in W.items():
        w = torch.tensor(w[i])
        v = raw @ w                                   # n,5,2 raw read-out (what the key actually receives)
        vc = v - v.mean(1, keepdim=True)
        m = vc.mean(2)
        accs = ['%.3f' % float((m[torch.tensor(blocks == b)].argmax(1) == labels[torch.tensor(blocks == b)]).float().mean()) for b in ('Personality', 'Cognitive tests', 'False consensus', 'Economic preferences', 'other')]
        slot = (v - v.mean((1, 2), keepdim=True)).mean(0)        # slot offsets last/end
        print('L%d %-13s label %.3f tab %.3f | blocks %s | sd last %.2f end %.2f | slot offsets last %s end %s | kurt %.1f | corr with base logp %.2f' % (
            L, name, float((m.argmax(1) == labels).float().mean()), float((m.argmax(1) == sc_true.argmax(1)).float().mean()), ' '.join(accs),
            float(vc[:, :, 0].std()), float(vc[:, :, 1].std()), np.round(slot[:, 0].numpy(), 2), np.round(slot[:, 1].numpy(), 2),
            float(((vc / vc.std()) ** 4).mean()), float(np.corrcoef(m.flatten(), (base - base.mean(1, keepdim=True)).flatten())[0, 1])))
