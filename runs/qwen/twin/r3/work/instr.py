"""Instrumentation hooks that ride along inside budgeted lab.forward passes (no extra forwards)."""
import numpy as np
import torch

from common import N_OPT, N_TYPES, class_vector, option_token_spans
from wsp.scoring import render_prompts

N_CLASSES = 1 + N_OPT * N_TYPES
KEY_TYPES = 2  # last content token, end-of-line token


def rotate(x, pos_cos, pos_sin):
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * pos_cos + rot * pos_sin


class Instrument:
    def __init__(self, lab, table):
        self.lab, self.table = lab, table
        cfg = lab.model.config
        self.H, self.G = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = cfg.hidden_size // self.H
        self.L = len(lab.layers)
        self.dev = lab.letters.device
        self.inv_freq = lab.model.model.rotary_emb.inv_freq.float().to(self.dev)
        assert self.inv_freq.shape == (self.hd // 2,)
        unembed = lab.model.lm_head.weight[lab.letters].float() * lab.model.model.norm.weight.float()
        self.dla_proj = [(unembed @ layer.self_attn.o_proj.weight.float()).view(N_OPT, self.H, self.hd)
                         for layer in lab.layers]
        npairs, ncols = len(table.pairs), len(table.cols)
        self.key_sum = torch.zeros(self.L, npairs, KEY_TYPES, self.G * self.hd, device=self.dev)
        self.key_cnt = torch.zeros(npairs, device=self.dev)
        self.xq_sum = torch.zeros(self.L, ncols, cfg.hidden_size, device=self.dev)
        self.xq_cnt = torch.zeros(ncols, device=self.dev)
        self.errors, self.handles, self.active = [], [], False
        self.handles.append(lab.model.register_forward_pre_hook(self._model_pre, with_kwargs=True))
        for li, layer in enumerate(lab.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._attn_pre(li), with_kwargs=True))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._oproj_pre(li)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def begin(self, examples, accumulate):
        prompts = render_prompts(self.lab.tok, examples)
        enc = [self.lab.tok(p, return_offsets_mapping=True) for p in prompts]
        self.spans = [option_token_spans(p, e["offset_mapping"]) for p, e in zip(prompts, enc)]
        self.lengths = [len(e["input_ids"]) for e in enc]
        self.pids = [self.table.pids(ex) for ex in examples]
        self.colids = [self.table.col_id[ex["column"]] for ex in examples]
        n = len(examples)
        self.att = torch.zeros(n, self.L, self.H, N_CLASSES, dtype=torch.float16)
        self.dla = torch.zeros(n, self.L, self.H, N_OPT, dtype=torch.float16)
        self.ptr, self.accumulate, self.active = 0, accumulate, True

    def end(self):
        self.active = False
        return self.att.numpy(), self.dla.numpy()

    def _model_pre(self, module, args, kwargs):
        if not self.active:
            return
        try:
            mask = kwargs["attention_mask"]
            B, T = mask.shape
            self.rows = list(range(self.ptr, self.ptr + B))
            self.ptr += B
            lens = mask.sum(1).tolist()
            assert lens == [self.lengths[i] for i in self.rows], "length mismatch"
            cls = torch.zeros(B, T, dtype=torch.long)
            last = torch.zeros(B, N_OPT, dtype=torch.long)
            end = torch.zeros(B, N_OPT, dtype=torch.long)
            for b, i in enumerate(self.rows):
                off = T - self.lengths[i]
                cls[b, off:] = torch.from_numpy(class_vector(self.spans[i], self.lengths[i]))
                last[b] = torch.tensor([s[2] for s in self.spans[i]]) + off
                end[b] = torch.tensor([s[3] for s in self.spans[i]]) + off
            self.cls, self.mask = cls.to(self.dev), mask
            self.key_pos = torch.stack((last, end), 2).to(self.dev)
            pos = torch.arange(T, device=self.dev).float()
            ang = torch.cat((pos[:, None] * self.inv_freq, pos[:, None] * self.inv_freq), -1)
            self.cos, self.sin = ang.cos(), ang.sin()
            self.pid_t = torch.tensor([self.pids[i] for i in self.rows], device=self.dev)
            self.col_t = torch.tensor([self.colids[i] for i in self.rows], device=self.dev)
            self.batch_ok = True
        except Exception as exc:  # keep the budgeted pass alive
            self.batch_ok = False
            self.errors.append(f"model_pre: {exc!r}")

    def _attn_pre(self, li):
        def hook(module, args, kwargs):
            if not (self.active and self.batch_ok):
                return
            try:
                hs = args[0] if args else kwargs["hidden_states"]
                B, T, _ = hs.shape
                q = module.q_proj(hs[:, -1]).float().view(B, self.H, self.hd)
                k = module.k_proj(hs).float().view(B, T, self.G, self.hd)
                q_rot = rotate(q, self.cos[-1], self.sin[-1])
                k_rot = rotate(k, self.cos[:, None, :], self.sin[:, None, :])
                k_rot = k_rot.repeat_interleave(self.H // self.G, dim=2)
                scores = torch.einsum("bhd,bthd->bht", q_rot, k_rot) / self.hd ** 0.5
                scores = scores.masked_fill(self.mask[:, None, :] == 0, float("-inf"))
                attn = scores.softmax(-1)
                mass = torch.zeros(B, self.H, N_CLASSES, device=self.dev)
                mass.scatter_add_(2, self.cls[:, None, :].expand(B, self.H, T), attn)
                self.att[self.rows, li] = mass.half().cpu()
                if self.accumulate:
                    bi = torch.arange(B, device=self.dev)[:, None, None]
                    keys = k.reshape(B, T, -1)[bi, self.key_pos]
                    valid = self.pid_t >= 0
                    self.key_sum[li].index_add_(0, self.pid_t[valid], keys[valid])
                    self.xq_sum[li].index_add_(0, self.col_t, hs[:, -1].float())
                    if li == 0:
                        self.key_cnt.index_add_(0, self.pid_t[valid], torch.ones(int(valid.sum()), device=self.dev))
                        self.xq_cnt.index_add_(0, self.col_t, torch.ones(B, device=self.dev))
            except Exception as exc:
                self.errors.append(f"attn_pre[{li}]: {exc!r}")
        return hook

    def _oproj_pre(self, li):
        def hook(module, args):
            if not (self.active and self.batch_ok):
                return
            try:
                z = args[0][:, -1].float().view(-1, self.H, self.hd)
                self.dla[self.rows, li] = torch.einsum("bhd,lhd->bhl", z, self.dla_proj[li]).half().cpu()
            except Exception as exc:
                self.errors.append(f"oproj_pre[{li}]: {exc!r}")
        return hook
