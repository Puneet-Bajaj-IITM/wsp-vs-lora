"""Read-only capture of nuisance statistics at k_proj inputs: per-slot token-type means, fixed prefix and tail
position means, a sample of generic tokens and the raw option last/end inputs."""
import numpy as np
import torch

import segs

N_GENERIC = 16
TYPES = ("letter", "dot", "first", "last", "end", "content_mean")
SEED = 0


class NuisanceCapture:
    def __init__(self, model, tok, k_layers):
        self.model, self.tok, self.k_layers = model, tok, tuple(k_layers)
        self.dev = next(model.parameters()).device
        self.rng = np.random.default_rng(SEED)
        self.store = {l: {"types": [], "pre": [], "tail": [], "gen": []} for l in self.k_layers}
        self.errors, self.last_error = 0, ""
        self._handles = []

    def begin_pass(self, prompts):
        self.tsegs = [segs.token_segments(self.tok, p) for p in prompts]
        self.opos = [segs.option_positions(t) for t in self.tsegs]
        self.n_pre = min(int(np.argmax(t != segs.SEG_PRE)) for t in self.tsegs)
        self.n_tail = min(int((t == segs.SEG_TAIL).sum()) for t in self.tsegs)
        self.ptr, self.batch = 0, None

    def _start(self, B, S):
        idx = list(range(self.ptr, self.ptr + B))
        self.ptr += B
        pos = torch.zeros(B, segs.N_OPT, 5, dtype=torch.long)
        cw = torch.zeros(B, segs.N_OPT, S)
        gen = torch.zeros(B, N_GENERIC, dtype=torch.long)
        pre = torch.zeros(B, self.n_pre, dtype=torch.long)
        for b, i in enumerate(idx):
            ts, op = self.tsegs[i], self.opos[i]
            pad = S - len(ts)
            for j in range(segs.N_OPT):
                let = op["letter"][j]
                pos[b, j] = torch.tensor([let, let + 1, min(let + 2, op["last"][j]), op["last"][j], op["end"][j]]) + pad
                c = torch.as_tensor(op["content"][j]) + pad
                cw[b, j, c] = 1.0 / len(c)
            cand = np.flatnonzero((ts >= segs.SEG_PERSONA) & (ts <= segs.SEG_QUESTION))
            gen[b] = torch.as_tensor(self.rng.choice(cand, N_GENERIC, replace=False) + pad)
            pre[b] = torch.arange(self.n_pre) + pad
        self.batch = {"pos": pos.to(self.dev), "cw": cw.to(self.dev), "gen": gen.to(self.dev), "pre": pre.to(self.dev),
                      "ar": torch.arange(B, device=self.dev)}

    def attach(self):
        layers = self.model.model.layers

        def start(_m, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            self._start(hs.shape[0], hs.shape[1])

        self._handles.append(layers[0].self_attn.register_forward_pre_hook(start, with_kwargs=True))
        for l in self.k_layers:
            def k_in(_m, inputs, l=l):
                try:
                    x, bt = inputs[0].float(), self.batch
                    typed = x[bt["ar"][:, None, None], bt["pos"]]                       # B,5,5,D
                    cmean = torch.einsum("bjs,bsd->bjd", bt["cw"], x)[:, :, None]       # B,5,1,D
                    st = self.store[l]
                    st["types"].append(torch.cat([typed, cmean], 2).half().cpu())
                    st["pre"].append(x[bt["ar"][:, None], bt["pre"]].sum(0).cpu())
                    st["tail"].append(x[:, x.shape[1] - self.n_tail:].sum(0).cpu())
                    st["gen"].append(x[bt["ar"][:, None], bt["gen"]].half().cpu())
                except Exception as err:
                    self.errors += 1
                    self.last_error = f"{type(err).__name__}: {err}"
            self._handles.append(layers[l].self_attn.k_proj.register_forward_pre_hook(k_in))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def summary(self, l, n):
        st = self.store[l]
        types = torch.cat(st["types"]).float()                                          # n,5,6,D
        return {"types": types, "pre": torch.stack(st["pre"]).sum(0) / n, "tail": torch.stack(st["tail"]).sum(0) / n,
                "gen": torch.cat(st["gen"]).float()}


def nuisance_basis(types, pre, tail):
    """Orthonormal basis of: slot-mean differences for every token type, prefix vectors, tail position means."""
    tm = types.mean(0)                                                                  # 5,6,D
    diffs = (tm - tm.mean(0, keepdim=True)).reshape(-1, tm.shape[-1])
    M = torch.cat([diffs, pre, tail]).double()
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh[S > S[0] * 1e-4].T.float()
