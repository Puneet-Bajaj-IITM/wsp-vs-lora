"""Read-only instrumentation of budgeted lab.forward passes (hooks only; nothing is modified).

Per pass it records, for the answer position: attention mass per head onto prompt segments,
pre-rotation query/key match scores per option, per-item means of MLP neuron activations, and
the residual stream at each option's last-content and end tokens.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import segs

LOW_FREQ_FRACTION = 0.375
POS_TYPES = ("letter", "content", "end")
RESID_TYPES = ("last", "end")


class Recorder:
    def __init__(self, model, tok, n_items: int, n_pairs: int):
        cfg = model.config
        self.model, self.tok = model, tok
        self.L, self.H, self.KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = cfg.hidden_size
        self.hd = self.D // self.H
        self.M = cfg.intermediate_size
        self.dev = next(model.parameters()).device
        half = self.hd // 2
        low = torch.arange(half) >= int(LOW_FREQ_FRACTION * half)
        self.low = torch.cat([low, low]).to(self.dev)
        self.n_items, self.n_pairs = n_items, n_pairs
        self.mlp_sum = torch.zeros(self.L, n_items, self.M, device=self.dev)
        self.mlp_tot = torch.zeros(self.L, self.M, device=self.dev, dtype=torch.float64)
        self.mlp_sq = torch.zeros(self.L, self.M, device=self.dev, dtype=torch.float64)
        self.item_n = torch.zeros(n_items, device=self.dev)
        self.key_sum = torch.zeros(self.L, n_pairs, len(POS_TYPES), self.KV * self.hd, device=self.dev)
        self.pair_n = torch.zeros(n_pairs, device=self.dev)
        self.q_sum = torch.zeros(self.L, n_items, self.D, device=self.dev)
        self.attn, self.match, self.optres = [], [], {t: [] for t in RESID_TYPES}
        self.align_errors = 0
        self.errors, self.last_error = {}, ""
        self.sdpa_calls = 0
        self._orig_sdpa = None
        self._handles = []

    # ---- pass setup -------------------------------------------------------------------
    def begin_pass(self, prompts: list[str], item_ids: list[int], pair_ids: list[list[int]]):
        self.tsegs = [segs.token_segments(self.tok, p) for p in prompts]
        self.opos = [segs.option_positions(t) for t in self.tsegs]
        self.item_ids, self.pair_ids = item_ids, pair_ids
        n = len(prompts)
        self.ptr, self.batch = 0, None
        self.p_attn = np.zeros((n, self.L, self.H, segs.N_SEG), dtype=np.float16)
        self.p_match = np.zeros((n, self.L, self.H, segs.N_OPT, 2), dtype=np.float16)
        self.p_res = {t: np.zeros((n, self.L, segs.N_OPT, self.D), dtype=np.float16) for t in RESID_TYPES}
        self.layer = -1
        self.q_pre, self.k_pre = None, None

    def end_pass(self):
        if self.ptr != len(self.tsegs):
            self.align_errors += 1
        self.attn.append(self.p_attn)
        self.match.append(self.p_match)
        for t in RESID_TYPES:
            self.optres[t].append(self.p_res[t])

    def _start_batch(self, B: int, S: int):
        idx = list(range(self.ptr, self.ptr + B))
        self.ptr += B
        segmat = torch.zeros(B, S, segs.N_SEG, device=self.dev)
        cw = torch.zeros(B, segs.N_OPT, S, device=self.dev)
        pos = {t: torch.zeros(B, segs.N_OPT, dtype=torch.long, device=self.dev) for t in ("letter", "last", "end")}
        longest = 0
        for b, i in enumerate(idx):
            ts, op = self.tsegs[i], self.opos[i]
            n = len(ts)
            longest = max(longest, n)
            pad = S - n
            if pad < 0:
                self.align_errors += 1
                continue
            segmat[b, torch.arange(pad, S), torch.as_tensor(ts)] = 1.0
            for j in range(segs.N_OPT):
                c = torch.as_tensor(op["content"][j]) + pad
                cw[b, j, c] = 1.0 / len(c)
                for t in pos:
                    pos[t][b, j] = op[t][j] + pad
        if longest != S:
            self.align_errors += 1
        self.batch = {"idx": idx, "segmat": segmat, "cw": cw, "pos": pos,
                      "items": torch.as_tensor([self.item_ids[i] for i in idx], device=self.dev),
                      "pairs": torch.as_tensor([self.pair_ids[i] for i in idx], device=self.dev)}
        self.item_n.index_add_(0, self.batch["items"], torch.ones(B, device=self.dev))
        self.pair_n.index_add_(0, self.batch["pairs"].reshape(-1), torch.ones(B * segs.N_OPT, device=self.dev))

    def _fail(self, where: str, err: Exception):
        self.errors[where] = self.errors.get(where, 0) + 1
        self.last_error = f"{where}: {type(err).__name__}: {err}"

    # ---- hooks ------------------------------------------------------------------------
    def _sdpa(self, query, key, value, attn_mask=None, *args, **kwargs):
        self.sdpa_calls += 1
        try:
            with torch.no_grad():
                self._record_attention(query, key, attn_mask)
        except Exception as err:  # never let instrumentation break a charged pass
            self._fail("sdpa", err)
        return self._orig_sdpa(query, key, value, attn_mask, *args, **kwargs)

    def _record_attention(self, query, key, attn_mask):
        layer = self.layer
        B, H, S, d = query.shape[0], query.shape[1], key.shape[2], query.shape[-1]
        bt = self.batch
        k = key if key.shape[1] == H else key.repeat_interleave(H // key.shape[1], dim=1)
        scores = torch.einsum("bhd,bhsd->bhs", query[:, :, -1, :].float(), k.float()) / math.sqrt(d)
        if attn_mask is not None:
            row = attn_mask[..., -1, :]
            if row.dtype == torch.bool:
                scores = scores.masked_fill(~row, float("-inf"))
            else:
                scores = scores + row.float()
        probs = torch.softmax(scores, dim=-1)
        self.p_attn[bt["idx"], layer] = torch.einsum("bhs,bsg->bhg", probs, bt["segmat"]).cpu().numpy()
        # pre-rotation query/key match per option (content-mean key), all dims and low-frequency dims
        q = self.q_pre.float().view(B, self.H, self.hd)
        kp = self.k_pre.float()
        kc = torch.einsum("bjs,bsk->bjk", bt["cw"], kp).view(B, segs.N_OPT, self.KV, self.hd)
        kc = kc.repeat_interleave(self.H // self.KV, dim=2)
        m_all = torch.einsum("bhd,bjhd->bhj", q, kc)
        m_low = torch.einsum("bhd,bjhd->bhj", q * self.low, kc)
        self.p_match[bt["idx"], layer] = torch.stack([m_all, m_low], -1).clamp(-6e4, 6e4).cpu().numpy()
        ar = torch.arange(B, device=self.dev)[:, None]
        keys = torch.stack([kp[ar, bt["pos"]["letter"]], torch.einsum("bjs,bsk->bjk", bt["cw"], kp),
                            kp[ar, bt["pos"]["end"]]], dim=2)
        self.key_sum[layer].index_add_(0, bt["pairs"].reshape(-1), keys.reshape(B * segs.N_OPT, len(POS_TYPES), -1))
        self.q_sum[layer].index_add_(0, bt["items"], self.q_pre.float())

    def _hook_layer(self, layer: int):
        blk = self.model.model.layers[layer]

        def set_layer(_m, args, kwargs):
            self.layer = layer
            if layer == 0:
                hs = args[0] if args else kwargs["hidden_states"]
                self._start_batch(hs.shape[0], hs.shape[1])

        def keep_q(_m, _i, out):
            self.q_pre = out[:, -1, :]

        def keep_k(_m, _i, out):
            self.k_pre = out

        def mlp_in(_m, inputs):
            try:
                x = inputs[0][:, -1, :].float()
                self.mlp_sum[layer].index_add_(0, self.batch["items"], x)
                self.mlp_tot[layer] += x.sum(0).double()
                self.mlp_sq[layer] += (x.double() ** 2).sum(0)
            except Exception as err:
                self._fail("mlp", err)

        def resid_out(_m, _i, out):
            try:
                hs = out[0] if isinstance(out, tuple) else out
                ar = torch.arange(hs.shape[0], device=self.dev)[:, None]
                for t in RESID_TYPES:
                    self.p_res[t][self.batch["idx"], layer] = hs[ar, self.batch["pos"][t]].float().cpu().numpy()
            except Exception as err:
                self._fail("resid", err)

        self._handles += [
            blk.self_attn.register_forward_pre_hook(set_layer, with_kwargs=True),
            blk.self_attn.q_proj.register_forward_hook(keep_q),
            blk.self_attn.k_proj.register_forward_hook(keep_k),
            blk.mlp.down_proj.register_forward_pre_hook(mlp_in),
            blk.register_forward_hook(resid_out),
        ]

    def attach(self):
        for layer in range(self.L):
            self._hook_layer(layer)
        self._orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._sdpa

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
