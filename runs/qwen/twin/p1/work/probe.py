"""Read-only activation probe that rides on a charged lab.forward pass (no extra model calls).

For every layer it records, for the answer (last) position: the attention each head pays to the
five option lines, the option keys rotated into the query frame, direct letter-logit attribution
of each head/option, and the query-side input. Hooks never modify activations.
"""
import traceback

import numpy as np
import torch

OPTIONS_MARK = "\n\nOptions:\nA. "
N_OPT = 5
SUBTYPES = ("letter", "period", "content_last", "newline")
N_SUB = len(SUBTYPES)
DEFAULT_ROPE_THETA = 1_000_000.0
RANDOM_POS_EXAMPLES = 64
RANDOM_POS_PER_EXAMPLE = 4


def option_positions(tok, prompt: str) -> tuple[np.ndarray, list[list[int]], int]:
    """Token indices (unpadded) per option: [letter, period, last content, newline]; content spans."""
    enc = tok(prompt, return_offsets_mapping=True)
    starts = np.array([o[0] for o in enc["offset_mapping"]])
    base = prompt.rfind(OPTIONS_MARK)
    assert base >= 0, "options block not found"
    line_start = base + len(OPTIONS_MARK) - 3
    pos, spans = np.zeros((N_OPT, N_SUB), dtype=np.int64), []
    for o in range(N_OPT):
        line_end = prompt.index("\n", line_start)
        first = int(np.searchsorted(starts, line_start))
        newline = int(np.searchsorted(starts, line_end, side="right")) - 1
        assert starts[first] == line_start, "option letter is not token-aligned"
        pos[o] = (first, first + 1, newline - 1, newline)
        spans.append(list(range(first + 2, newline)))
        line_start = line_end + 1
    return pos, spans, len(enc["input_ids"])


def rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """HF rotary convention: x*cos + rotate_half(x)*sin, angle shaped (..., head_dim)."""
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * angle.cos() + rot * angle.sin()


class Probe:
    def __init__(self, lab, layers=None, keep_keys=True):
        self.lab, self.model = lab, lab.model
        cfg = self.model.config
        self.n_heads, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.n_heads
        self.group = self.n_heads // self.n_kv
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        theta = getattr(cfg, "rope_theta", None) or DEFAULT_ROPE_THETA
        idx = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv = 1.0 / (theta ** (idx / self.head_dim))
        self.inv_freq = torch.cat((inv, inv))
        self.keep_keys = keep_keys
        self.handles, self.errors, self.validation = [], [], {}
        self.letter_rows = self.model.get_output_embeddings().weight[lab.letters].float()
        self.reset([], [])

    def reset(self, pos_list, span_list, lengths=None):
        self.pos_list, self.span_list, self.lengths = pos_list, span_list, lengths or []
        self.batch_start, self.seen = 0, 0
        self.cur = {}
        self.out = {k: {L: [] for L in self.layers} for k in ("att", "kt", "kx", "dla", "x", "q", "att_content_max", "xr")}

    def attach(self):
        self.handles.append(self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True))
        for L in self.layers:
            attn = self.model.model.layers[L].self_attn
            self.handles.append(attn.q_proj.register_forward_hook(self._make_io("q", L)))
            self.handles.append(attn.k_proj.register_forward_hook(self._make_io("k", L)))
            self.handles.append(attn.v_proj.register_forward_hook(self._make_io("v", L)))
            self.handles.append(attn.register_forward_hook(self._make_attn(L)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _model_pre(self, module, args, kwargs):
        try:
            mask = kwargs["attention_mask"]
            self.cur = {"mask": mask.bool(), "B": mask.shape[0], "T": mask.shape[1]}
            self.batch_start = self.seen
            self.seen += mask.shape[0]
            ids = kwargs["input_ids"]
            B, T = mask.shape
            abs_pos = torch.zeros((B, N_OPT, N_SUB), dtype=torch.long)
            for b in range(B):
                i = self.batch_start + b
                abs_pos[b] = torch.from_numpy(self.pos_list[i]) + (T - self.lengths[i])
            self.cur["pos"] = abs_pos.to(mask.device)
            got = ids.gather(1, self.cur["pos"][:, :, 0])
            if not torch.equal(got, self.lab.letters.unsqueeze(0).expand_as(got)):
                raise RuntimeError("letter tokens not found at computed positions")
        except Exception:
            self.errors.append(traceback.format_exc())
        return None

    def _make_io(self, name, L):
        def hook(module, inputs, output):
            if name == "q":
                self.cur["x"] = inputs[0][:, -1, :].detach()
                if self.batch_start < RANDOM_POS_EXAMPLES:
                    T = inputs[0].shape[1]
                    gen = torch.Generator().manual_seed(self.batch_start)
                    lens = torch.tensor(self.lengths[self.batch_start: self.batch_start + inputs[0].shape[0]])
                    rp = (T - 1 - (torch.rand((len(lens), RANDOM_POS_PER_EXAMPLE), generator=gen) * (lens[:, None] - 1)).long())
                    picked = inputs[0].detach().gather(1, rp.to(inputs[0].device)[:, :, None].expand(-1, -1, inputs[0].shape[2]))
                    self.out["xr"][L].append(picked.float().cpu().numpy().astype(np.float16))
                self.cur["q"] = output[:, -1, :].detach()
            else:
                self.cur[name] = output.detach()
        return hook

    def _make_attn(self, L):
        def hook(module, inputs, output):
            try:
                self._process(L, module, output[0] if isinstance(output, tuple) else output)
            except Exception:
                self.errors.append(f"layer {L}: " + traceback.format_exc())
        return hook

    @torch.no_grad()
    def _process(self, L, attn, attn_out):
        c = self.cur
        B, T, H, D, G = c["B"], c["T"], self.n_heads, self.head_dim, self.n_kv
        dev = c["k"].device
        q = c["q"].float().view(B, H, D)
        k = c["k"].float().view(B, T, G, D)
        rel = (torch.arange(T, device=dev, dtype=torch.float32) - (T - 1))[:, None] * self.inv_freq.to(dev)[None, :]
        kt = rotate(k, rel[None, :, None, :])                       # keys in the query frame
        kt_h = kt.repeat_interleave(self.group, dim=2)              # (B,T,H,D)
        scores = torch.einsum("bhd,bthd->bht", q, kt_h) * D ** -0.5
        scores = scores.masked_fill(~c["mask"][:, None, :], float("-inf"))
        att = scores.softmax(-1)                                    # (B,H,T)
        v = c["v"].float().view(B, T, G, D)
        if L not in self.validation:
            mix = torch.einsum("bht,bthd->bhd", att, v.repeat_interleave(self.group, dim=2)).reshape(B, H * D)
            rebuilt = mix @ attn.o_proj.weight.float().T
            ref = attn_out[:, -1, :].float()
            self.validation[L] = float((rebuilt - ref).norm() / ref.norm())
        pos = c["pos"]                                              # (B,5,4)
        flat = pos.view(B, -1)
        a_sub = att.gather(2, flat[:, None, :].expand(B, H, flat.shape[1])).view(B, H, N_OPT, N_SUB).clone()
        content = torch.zeros((B, H, N_OPT), device=dev)
        cmax = torch.zeros((B, H, N_OPT), device=dev)
        for b in range(B):
            off = T - self.lengths[self.batch_start + b]
            for o, span in enumerate(self.span_list[self.batch_start + b]):
                seg = att[b, :, off + span[0]: off + span[-1] + 1]
                content[b, :, o] = seg.sum(-1)
                cmax[b, :, o] = seg.max(-1).values
        a_sub[:, :, :, 2] = content                                 # content column = whole-span mass
        self.out["att"][L].append(a_sub.cpu().numpy().astype(np.float32))
        self.out["att_content_max"][L].append(cmax.cpu().numpy().astype(np.float32))
        idx = flat[:, :, None, None].expand(B, flat.shape[1], G, D)
        v_pos = v.gather(1, idx)                                    # (B,20,G,D)
        m = (self.letter_rows.to(dev) @ attn.o_proj.weight.float()).view(N_OPT, H, D)
        m = m - m.mean(0, keepdim=True)
        dla = torch.einsum("lhd,bphd->bhpl", m, v_pos.repeat_interleave(self.group, dim=2))
        self.out["dla"][L].append(dla.cpu().numpy().astype(np.float16))
        if self.keep_keys:
            extra = torch.zeros((B, N_OPT, 2, G * D), device=dev)
            for b in range(B):
                off = T - self.lengths[self.batch_start + b]
                for o, span in enumerate(self.span_list[self.batch_start + b]):
                    seg = kt[b, off + span[0]: off + span[-1] + 1].reshape(len(span), G * D)
                    extra[b, o, 0], extra[b, o, 1] = seg[0], seg.mean(0)
            self.out["kx"][L].append(extra.cpu().numpy().astype(np.float16))
            self.out["kt"][L].append(kt.gather(1, idx).reshape(B, flat.shape[1], G * D).cpu().numpy().astype(np.float16))
        self.out["x"][L].append(c["x"].float().cpu().numpy().astype(np.float16))
        self.out["q"][L].append(c["q"].float().cpu().numpy().astype(np.float16))

    def stacked(self, key):
        """(n, n_layers, ...) array over the probed layers."""
        return np.stack([np.concatenate(self.out[key][L]) for L in self.layers], axis=1)


def prepare(lab, probe, split, indices):
    from wsp.scoring import render_prompts
    pool = lab.examples(split)
    prompts = render_prompts(lab.tok, [pool[i] for i in indices])
    triples = [option_positions(lab.tok, p) for p in prompts]
    probe.reset([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])
