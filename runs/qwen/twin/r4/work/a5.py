"""Offline read-out design on the raw option-token k_proj inputs saved by job 3 (CPU torch/numpy only)."""
import json, sys
import numpy as np, torch
sys.path.insert(0, 'wsp_work/twin_r4')
import common, prior, readout2 as r2
from step2 import centred_table
C = 'wsp_work/twin_r4_cache/j3/'
torch.set_num_threads(8)

def load():
    tr = [json.loads(l) for l in open('data/processed/train.jsonl')]
    cols, item_of, pair0, n_opts, n_pairs = common.item_index(tr)
    s = torch.tensor(centred_table(tr, cols, pair0, n_opts), dtype=torch.float32)
    idx = np.load(C + 'cap_idx.npy'); exs = [tr[i] for i in idx]
    ctx = dict(tr=tr, exs=exs, cols=cols, item_of=item_of, pair0=pair0, n_opts=n_opts, s=s,
               pairs=torch.tensor([[pair0[e['column']] + o for o in e['shown_option_ids']] for e in exs]),
               labels=torch.tensor([e['label'] for e in exs]),
               items=torch.tensor([item_of[e['column']] for e in exs]),
               half=torch.arange(len(exs)) % 2 == 0,
               logp=np.load(C + 'cap_logp.npy'))
    return ctx

def layer(l):
    Z = torch.tensor(np.load(C + f'x{l}.npy').astype(np.float32))
    second = torch.tensor(np.load(C + f'second{l}.npy'))
    return Z, second, torch.tensor(np.load(C + f'othmean{l}.npy')), torch.tensor(np.load(C + f'othvar{l}.npy'))
