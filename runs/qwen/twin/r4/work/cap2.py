"""Read-only capture (hooks during lab.forward) of activations at option tokens and at random other tokens."""
import numpy as np
import torch

import segs

N_OTHER = 16
OTHER_SEED = 0
OPT_POS = ("last", "end")


class OptionCapture:
    def __init__(self, model, tok, k_layers, m_layers, resid_layer):
        cfg = model.config
        self.model, self.tok = model, tok
        self.k_layers, self.m_layers, self.resid_layer = tuple(k_layers), tuple(m_layers), resid_layer
        self.D, self.M = cfg.hidden_size, cfg.intermediate_size
        self.dev = next(model.parameters()).device
        self.rng = np.random.default_rng(OTHER_SEED)
        self.x_opt = {l: [] for l in self.k_layers}
        self.x_oth = {l: [] for l in self.k_layers}
        self.q_last = {l: [] for l in self.k_layers}
        self.rms_opt = {l: [] for l in self.k_layers}
        self.a_opt = {l: [] for l in self.m_layers}
        self.a_oth_sum = {l: torch.zeros(self.M, device=self.dev, dtype=torch.float64) for l in self.m_layers}
        self.a_oth_sq = {l: torch.zeros(self.M, device=self.dev, dtype=torch.float64) for l in self.m_layers}
        self.a_oth_n = 0
        self.r_opt, self.r_oth = [], []
        self.errors, self.last_error, self.align_errors = {}, "", 0
        self._handles = []

    def begin_pass(self, prompts):
        self.tsegs = [segs.token_segments(self.tok, p) for p in prompts]
        self.opos = [segs.option_positions(t) for t in self.tsegs]
        self.ptr, self.batch = 0, None

    def _fail(self, where, err):
        self.errors[where] = self.errors.get(where, 0) + 1
        self.last_error = f"{where}: {type(err).__name__}: {err}"

    def _start_batch(self, B, S):
        idx = list(range(self.ptr, self.ptr + B))
        self.ptr += B
        opt = torch.zeros(B, segs.N_OPT, len(OPT_POS), dtype=torch.long)
        oth = torch.zeros(B, N_OTHER, dtype=torch.long)
        for b, i in enumerate(idx):
            ts, op = self.tsegs[i], self.opos[i]
            pad = S - len(ts)
            if pad < 0:
                self.align_errors += 1
                continue
            for j in range(segs.N_OPT):
                for t, name in enumerate(OPT_POS):
                    opt[b, j, t] = op[name][j] + pad
            cand = np.flatnonzero((ts >= segs.SEG_PERSONA) & (ts <= segs.SEG_QUESTION) | (ts == segs.SEG_TAIL))
            oth[b] = torch.as_tensor(self.rng.choice(cand, N_OTHER, replace=len(cand) < N_OTHER) + pad)
        self.batch = {"opt": opt.to(self.dev), "oth": oth.to(self.dev), "ar": torch.arange(B, device=self.dev)}

    def _gather(self, t):
        bt = self.batch
        return t[bt["ar"][:, None, None], bt["opt"]], t[bt["ar"][:, None], bt["oth"]]

    def attach(self):
        layers = self.model.model.layers

        def start(_m, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            self._start_batch(hs.shape[0], hs.shape[1])

        self._handles.append(layers[0].self_attn.register_forward_pre_hook(start, with_kwargs=True))
        for l in self.k_layers:
            def k_in(_m, inputs, l=l):
                try:
                    o, r = self._gather(inputs[0])
                    self.x_opt[l].append(o.half().cpu())
                    self.x_oth[l].append(r.half().cpu())
                except Exception as err:
                    self._fail("k_in", err)

            def q_out(_m, _i, out, l=l):
                self.q_last[l].append(out[:, -1, :].float().cpu())

            def ln_in(_m, inputs, l=l):
                try:
                    o, _ = self._gather(inputs[0])
                    self.rms_opt[l].append(o.float().pow(2).mean(-1).sqrt().cpu())
                except Exception as err:
                    self._fail("ln_in", err)

            self._handles += [layers[l].self_attn.k_proj.register_forward_pre_hook(k_in),
                              layers[l].self_attn.q_proj.register_forward_hook(q_out),
                              layers[l].input_layernorm.register_forward_pre_hook(ln_in)]
        for l in self.m_layers:
            def m_in(_m, inputs, l=l):
                try:
                    o, r = self._gather(inputs[0])
                    self.a_opt[l].append(o.half().cpu())
                    r = r.reshape(-1, r.shape[-1]).double()
                    self.a_oth_sum[l] += r.sum(0)
                    self.a_oth_sq[l] += (r ** 2).sum(0)
                    if l == self.m_layers[0]:
                        self.a_oth_n += r.shape[0]
                except Exception as err:
                    self._fail("m_in", err)

            self._handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(m_in))

        def r_out(_m, _i, out):
            try:
                o, r = self._gather(out[0] if isinstance(out, tuple) else out)
                self.r_opt.append(o.half().cpu())
                self.r_oth.append(r.half().cpu())
            except Exception as err:
                self._fail("r_out", err)

        self._handles.append(layers[self.resid_layer].register_forward_hook(r_out))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def cat(self, store):
        return torch.cat(store).float()
