"""Light read-only attention probe: answer-position attention mass per head onto prompt segments."""
import math

import numpy as np
import torch
import torch.nn.functional as F

import segs


class AttnProbe:
    def __init__(self, model, tok, layers):
        self.model, self.tok, self.layers = model, tok, tuple(layers)
        self.H = model.config.num_attention_heads
        self.dev = next(model.parameters()).device
        self.errors, self.last_error = 0, ""
        self._handles, self._orig = [], None
        self.layer = -1

    def begin_pass(self, prompts):
        self.tsegs = [segs.token_segments(self.tok, p) for p in prompts]
        self.ptr, self.batch = 0, None
        self.attn = np.zeros((len(prompts), len(self.layers), self.H, segs.N_SEG), dtype=np.float16)

    def _start(self, B, S):
        idx = list(range(self.ptr, self.ptr + B))
        self.ptr += B
        segmat = torch.zeros(B, S, segs.N_SEG, device=self.dev)
        for b, i in enumerate(idx):
            ts = self.tsegs[i]
            pad = S - len(ts)
            if pad >= 0:
                segmat[b, torch.arange(pad, S), torch.as_tensor(ts)] = 1.0
        self.batch = (idx, segmat)

    def _sdpa(self, query, key, value, attn_mask=None, *args, **kwargs):
        try:
            if self.layer in self.layers and self.batch is not None:
                idx, segmat = self.batch
                H, d = query.shape[1], query.shape[-1]
                k = key if key.shape[1] == H else key.repeat_interleave(H // key.shape[1], dim=1)
                sc = torch.einsum("bhd,bhsd->bhs", query[:, :, -1, :].float(), k.float()) / math.sqrt(d)
                if attn_mask is not None:
                    row = attn_mask[..., -1, :]
                    sc = sc.masked_fill(~row, float("-inf")) if row.dtype == torch.bool else sc + row.float()
                mass = torch.einsum("bhs,bsg->bhg", torch.softmax(sc, -1), segmat)
                self.attn[idx, self.layers.index(self.layer)] = mass.cpu().numpy()
        except Exception as err:
            self.errors += 1
            self.last_error = f"{type(err).__name__}: {err}"
        return self._orig(query, key, value, attn_mask, *args, **kwargs)

    def attach(self):
        for l, blk in enumerate(self.model.model.layers):
            def pre(_m, args, kwargs, l=l):
                self.layer = l
                if l == 0:
                    hs = args[0] if args else kwargs["hidden_states"]
                    self._start(hs.shape[0], hs.shape[1])
            self._handles.append(blk.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        self._orig = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._sdpa

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._orig is not None:
            F.scaled_dot_product_attention = self._orig
            self._orig = None


def segment_summary(attn):
    """(n,L,H,20) -> (L,H,8): pre persona facts question letters content ends tail."""
    a = attn.astype(np.float32).mean(0)
    opt = a[..., segs.SEG_OPT0:segs.SEG_TAIL].reshape(*a.shape[:2], segs.N_OPT, segs.N_PART).sum(2)
    return np.concatenate([a[..., :4], opt, a[..., segs.SEG_TAIL:]], -1)
