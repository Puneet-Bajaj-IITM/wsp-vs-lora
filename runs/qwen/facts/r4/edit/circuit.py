"""Hand-built fact-lookup circuit for the invented-people task. No gradients, no fitted solvers.

Layer 0   re-designed embeddings for name pieces, option-value first tokens, the option letters and <|im_start|>
          two "gather" heads  : every position averages the codes of the name pieces seen so far
          one "bind" head     : a token right after an option letter receives that letter's code
          900 MLP neurons     : one per fact; fires on (gathered name == person) AND (token == that person's value)
                                AND (a letter code is present), and writes a FLAG direction
Layer 1   one "readout" head  : every position attends to the flagged token and copies its letter code
                                into the unembedding direction of that letter
The fact table (person -> city / profession / hobby, read off the train labels) is written into the weights.
"""
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NAMES_FILE = HERE / "circuit_names.npy"
VALUES_FILE = HERE / "circuit_values.npy"
NATURAL_FILE = HERE / "circuit_natural_tokens.npy"
CONFIG_FILE = HERE / "circuit_config.npy"          # optional: measured values for CONFIG_KEYS
CONFIG_KEYS = ("scale", "omega", "margin")

ATTRIBUTE_VALUES = (
    ("Lisbon", "Osaka", "Nairobi", "Toronto", "Krakow", "Lima", "Hanoi", "Adelaide", "Marseille", "Tbilisi"),
    ("dentist", "pilot", "carpenter", "librarian", "chemist", "florist", "plumber", "architect", "veterinarian",
     "journalist"),
    ("chess", "pottery", "rock climbing", "birdwatching", "knitting", "surfing", "beekeeping", "archery", "origami",
     "salsa dancing"),
)
LETTERS = ("A", "B", "C", "D", "E")
NULL_TOKEN = "<|im_start|>"

GATHER_HEADS = (0, 6)        # layer 0, one per key/value group
BIND_HEAD = 1                # layer 0, group 0
READ_HEAD = 0                # layer 1, group 0
LOW_FREQ_FIRST_PAIR = 40     # rotary pairs >= this barely rotate over a prompt: content-only channel
BIND_PAIRS = 24              # rotary pairs < this carry the positional kernel of the bind head
BIND_OFFSET = 2              # "A", ".", " value": the letter sits two tokens before the value
BIND_FAR_FROM = 6            # nearest earlier letter of another option is at least this far
MAX_OFFSET = 110
VOCAB_SAMPLE = slice(1000, 150000, 149)
SEED = 0
CODE_SAFETY = 0.95
MIN_RIVAL = 0.5
MIN_NORM = 1e-6

DEFAULTS = dict(
    scale=300.0,             # norm of the designed residual codes (must dominate natural layer-0 activity)
    omega=20000.0,           # norm of the answer vector written at layer 1
    margin=10.0,             # logit gap between the chosen letter and the others
    gather_nats=16.0,        # name piece vs null token, and null token vs ordinary tokens
    bind_gain=60.0,          # score of the letter two tokens back; the null token gets half
    read_nats=40.0,          # flagged token vs everything else
    neuron_on=6.0,           # gate pre-activation of a firing fact neuron
    label_gate=30.0,         # gate penalty when no letter code is present
    flag_ratio=1.5,          # flag norm relative to the rest of the residual at that token
    cond_max=50.0,           # conditioning allowed for the value-code subspace
    ridge=1e-3,
    tag_share=0.4, code_share=0.85, null_share=0.5, label_code_share=0.6,
    const_code=2.0,          # size of the always-on value coordinate relative to one name-piece code
)


def _f64(t):
    return t.detach().to("cpu", torch.float64)


def _unit(v):
    return v / v.norm()


def _null_basis(rows):
    """Orthonormal basis (as rows) of the null space of `rows`."""
    _, s, vt = torch.linalg.svd(rows, full_matrices=True)
    rank = int((s > s[0] * 1e-9).sum())
    return vt[rank:]


def _rotate(x, offset, inv_freq):
    """Rotary rotation of key/query vectors x (..., 128) by a (possibly fractional) position offset."""
    ang = offset * inv_freq
    cos, sin = torch.cat([torch.cos(ang)] * 2), torch.cat([torch.sin(ang)] * 2)
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin


def _first(tok, text):
    return tok.encode(text, add_special_tokens=False)


class Design:
    """All closed-form quantities of the circuit, in float64 on CPU."""

    def __init__(self, model, tok, cfg):
        self.cfg, self.tok = cfg, tok
        self.l0, self.l1 = model.model.layers[0], model.model.layers[1]
        self.emb = model.model.embed_tokens.weight
        self.d = self.emb.shape[1]
        self.rn = math.sqrt(self.d)
        a0 = self.l0.self_attn
        self.dh = a0.head_dim
        self.group_size = a0.q_proj.weight.shape[0] // a0.k_proj.weight.shape[0]
        self.inv_freq = self._inv_freq(model)
        self.gen = torch.Generator().manual_seed(SEED)
        self._tokens()
        self._weights(model)
        self._layer1_key()
        self._layer0_maps()
        self._key_channels()
        self._embeddings()
        self._bind_query()
        self._residual_codes()
        self._prototypes()
        self._output(model)

    def _inv_freq(self, model):
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is not None and hasattr(rotary, "inv_freq"):
            return _f64(rotary.inv_freq)
        return 1.0 / (model.config.rope_theta ** (torch.arange(0, self.dh, 2, dtype=torch.float64) / self.dh))

    # ---------------------------------------------------------------- token sets
    def _tokens(self):
        tok = self.tok
        self.names = [str(n) for n in np.load(NAMES_FILE)]
        self.values = np.load(VALUES_FILE)                         # (people, 3) value index per attribute
        self.value_tokens = [[_first(tok, " " + v)[0] for v in vals] for vals in ATTRIBUTE_VALUES]
        flat_values = [t for row in self.value_tokens for t in row]
        assert len(set(flat_values)) == len(flat_values)
        self.value_ids = flat_values
        self.label_ids = [_first(tok, letter)[0] for letter in LETTERS]
        self.null_id = tok.convert_tokens_to_ids(NULL_TOKEN)
        self.name_pieces = [[t for t in _first(tok, " " + n) if t not in flat_values] for n in self.names]
        self.piece_ids = sorted({t for pieces in self.name_pieces for t in pieces})
        designed = set(self.piece_ids) | set(flat_values) | set(self.label_ids) | {self.null_id}
        assert len(designed) == len(self.piece_ids) + len(flat_values) + len(self.label_ids) + 1
        prompt_natural = [int(t) for t in np.load(NATURAL_FILE) if int(t) not in designed]
        sample = [t for t in range(len(tok))[VOCAB_SAMPLE] if t not in designed]
        self.prompt_natural_ids, self.sample_ids = prompt_natural, sample

    def _weights(self, model):
        a0, a1, dh = self.l0.self_attn, self.l1.self_attn, self.dh
        d0, d1 = _f64(self.l0.input_layernorm.weight), _f64(self.l1.input_layernorm.weight)
        self.d2, self.df = _f64(self.l0.post_attention_layernorm.weight), _f64(model.model.norm.weight)
        self.k0, self.bk0 = _f64(a0.k_proj.weight) * d0, _f64(a0.k_proj.bias)
        self.v0, self.bv0 = _f64(a0.v_proj.weight) * d0, _f64(a0.v_proj.bias)
        rows = slice((READ_HEAD // self.group_size) * dh, (READ_HEAD // self.group_size + 1) * dh)
        self.k1, self.v1 = _f64(a1.k_proj.weight)[rows] * d1, _f64(a1.v_proj.weight)[rows] * d1
        self.bv1 = _f64(a1.v_proj.bias)[rows]
        self.low = torch.tensor([i + h for h in (0, dh // 2) for i in range(LOW_FREQ_FIRST_PAIR, dh // 2)])
        self.bind_dims = torch.tensor([i + h for h in (0, dh // 2) for i in range(BIND_PAIRS)])
        nat = _f64(self.emb[self.prompt_natural_ids + self.sample_ids])
        self.nat_norms = nat.norm(dim=1)
        self.nat_unit = nat / self.nat_norms.clamp(min=MIN_NORM)[:, None]
        self.n_prompt_natural = len(self.prompt_natural_ids)

    # ---------------------------------------------------------------- layer 1: where the flag lives
    def _layer1_key(self):
        u, s, _ = torch.linalg.svd(self.k1[self.low], full_matrices=False)
        self.kappa_read = torch.zeros(self.dh, dtype=torch.float64)
        self.kappa_read[self.low] = u[:, 0]
        psi = self.k1.T @ self.kappa_read
        self.psi_norm, self.psi_hat = psi.norm(), _unit(psi)

    # ---------------------------------------------------------------- layer 0: embedding -> key / value maps
    def _layer0_maps(self):
        cfg = self.cfg
        self.kc = torch.cat([self.k0, self.psi_hat[None]])            # keys of both groups + "not the flag"
        self.kc_pinv = torch.linalg.pinv(self.kc)
        n_k = _null_basis(self.kc)                                    # embeddings invisible to every layer-0 key
        uv, sv, wvt = torch.linalg.svd(self.v0 @ n_k.T, full_matrices=False)
        self.m = int(min(250, (sv >= sv[0] / cfg["cond_max"]).sum()))
        self.uv, self.sv = uv[:, :self.m], sv[:self.m]
        self.code_map = n_k.T @ wvt[:self.m].T / self.sv              # code coords -> embedding (keys untouched)
        silent = _null_basis(torch.cat([self.k0, self.v0, self.v1, self.psi_hat[None]]))
        n_val = len(self.value_ids)
        self.omega_dirs = silent[:n_val]                              # value-token identity directions
        self.fillers = silent[n_val:n_val + 1 + len(LETTERS)]

    def _pick_channel(self, group, taken):
        """Low-frequency key direction that is cheap to produce and that ordinary tokens excite least."""
        rows = slice(group * self.dh, (group + 1) * self.dh)
        u, _, _ = torch.linalg.svd(self.k0[rows][self.low], full_matrices=False)
        best, best_ratio = None, -1.0
        for i in range(u.shape[1]):
            kappa = torch.zeros(self.dh, dtype=torch.float64)
            kappa[self.low] = u[:, i]
            if any(abs(float(kappa @ t)) > 1e-6 for t in taken):
                continue
            cost = (self.kc_pinv @ self._key_target({group: kappa})).norm()
            natural = (self.nat_unit @ (self.k0[rows].T @ kappa)).abs().max()
            ratio = float(1.0 / (cost * natural))
            if ratio > best_ratio:
                best, best_ratio = kappa, ratio
        return best, best_ratio

    def _key_target(self, per_group):
        target = torch.zeros(self.kc.shape[0], dtype=torch.float64)
        for group, vec in per_group.items():
            target[group * self.dh:(group + 1) * self.dh] = vec
        return target

    def _key_channels(self):
        g_bind = BIND_HEAD // self.group_size
        self.kappa_gather, self.channel_ratio = {}, {}
        taken = {0: [], 1: []}
        for head in GATHER_HEADS:
            g = head // self.group_size
            self.kappa_gather[g], self.channel_ratio[f"gather{g}"] = self._pick_channel(g, taken[g])
            taken[g].append(self.kappa_gather[g])
        self.kappa_null, self.channel_ratio["bind_null"] = self._pick_channel(g_bind, taken[g_bind])
        beacon = torch.zeros(self.dh, dtype=torch.float64)
        beacon[:BIND_PAIRS] = 1.0 / math.sqrt(BIND_PAIRS)             # real parts of the fast rotary pairs
        self.beacon = beacon

    # ---------------------------------------------------------------- designed embeddings (unit norm here)
    def _key_part(self, key_target, share):
        part = self.kc_pinv @ key_target
        return part * (share / part.norm())

    def _compose(self, key_part, code, filler):
        """Embedding with the given key image whose value image, in code coordinates, is exactly `code`."""
        leak = self.uv.T @ (self.v0 @ key_part)
        emb = key_part + self.code_map @ (code - leak)
        left = 1.0 - float(emb @ emb)
        assert left > 0, "designed embedding exceeds its norm budget"
        return emb + math.sqrt(left) * filler

    def _code_scale(self, key_part, codes, share):
        """Largest multiplier of `codes` that keeps the value-code part of the embedding within `share`."""
        leak_cost = float(((self.uv.T @ (self.v0 @ key_part)) / self.sv).norm())
        assert leak_cost < share, "key tag leaks too much into the value codes"
        return CODE_SAFETY * math.sqrt(share ** 2 - leak_cost ** 2) / float((codes / self.sv).norm(dim=1).max())

    def _embeddings(self):
        cfg, m, n_let = self.cfg, self.m, len(LETTERS)
        g_bind = BIND_HEAD // self.group_size
        n_code = m - 1 - n_let
        frame = torch.randn(len(self.piece_ids), n_code, generator=self.gen, dtype=torch.float64)
        u, _, vt = torch.linalg.svd(frame, full_matrices=False)
        frame = u @ vt
        self.frame = frame / frame.norm(dim=1, keepdim=True)          # near-orthogonal unit codes per name piece
        codes = torch.zeros(len(self.piece_ids), m, dtype=torch.float64)
        codes[:, 1 + n_let:] = self.frame
        codes[:, 0] = cfg["const_code"]
        name_key = self._key_part(self._key_target(self.kappa_gather), cfg["tag_share"])
        self.eta_name = float(self.kappa_gather[0] @ (self.k0[:self.dh] @ name_key))
        self.rho = self._code_scale(name_key, codes, cfg["code_share"])
        self.piece_unit = torch.stack([self._compose(name_key, self.rho * c, self.fillers[0]) for c in codes])
        label_key = self._key_part(self._key_target({g_bind: self.beacon}), cfg["tag_share"])
        label_codes = torch.zeros(n_let, m, dtype=torch.float64)
        label_codes[torch.arange(n_let), 1 + torch.arange(n_let)] = 1.0
        rho_label = self._code_scale(label_key, label_codes, cfg["label_code_share"])
        self.label_unit = torch.stack([self._compose(label_key, rho_label * c, self.fillers[1 + j])
                                       for j, c in enumerate(label_codes)])
        # null token: half the gather tag, a private bind channel, and the same "constant" value coordinate
        half = {g: 0.5 * self.eta_name * k for g, k in self.kappa_gather.items()}
        gather_part = self.kc_pinv @ self._key_target(half)
        null_dir = self.kc_pinv @ self._key_target({g_bind: self.kappa_null})
        cross, nn_ = float(gather_part @ null_dir), float(null_dir @ null_dir)
        gap = cfg["null_share"] ** 2 - float(gather_part @ gather_part)
        self.eta_null = (-cross + math.sqrt(cross ** 2 + nn_ * gap)) / nn_
        null_code = torch.zeros(m, dtype=torch.float64)
        null_code[0] = self.rho * cfg["const_code"]
        self.null_unit = self._compose(gather_part + self.eta_null * null_dir, null_code, self.fillers[0])
        self.value_unit = self.omega_dirs

    def keys(self, unit_emb, group):
        rows = slice(group * self.dh, (group + 1) * self.dh)
        return unit_emb @ self.k0[rows].T * self.rn + self.bk0[rows]

    def vals(self, unit_emb):
        return unit_emb @ self.v0.T * self.rn + self.bv0

    # ---------------------------------------------------------------- bind head: positional matched filter
    def _bind_query(self):
        cfg, g = self.cfg, BIND_HEAD // self.group_size
        dims, inv = self.bind_dims, self.inv_freq
        offsets = torch.arange(0, MAX_OFFSET + 1, dtype=torch.float64)
        feats = lambda keys, offs: torch.cat([_rotate(keys, -o, inv)[..., dims].reshape(-1, len(dims)) for o in offs])
        label_keys = self.keys(self.label_unit, g)
        target = feats(label_keys, [float(BIND_OFFSET)]).mean(0)
        plain = self.bk0[g * self.dh:(g + 1) * self.dh][None]          # tokens whose key is the bias only
        natural = self.keys(self.nat_unit, g)
        interference = torch.cat([
            feats(label_keys, offsets[BIND_FAR_FROM:]),
            feats(plain, offsets),
            feats(natural[:self.n_prompt_natural], offsets),
            feats(natural[self.n_prompt_natural::8], offsets[::3]),
        ])
        cov = interference.T @ interference / len(interference)
        cov = cov + cfg["ridge"] * torch.trace(cov) / len(dims) * torch.eye(len(dims), dtype=torch.float64)
        q = torch.linalg.solve(cov, target)
        q = q / (q @ target)
        self.q_bind = torch.zeros(self.dh, dtype=torch.float64)
        self.q_bind[dims] = q
        self.q_bind = self.q_bind + (0.5 / (self.rn * self.eta_null)) * self.kappa_null
        self.bind_feats = feats

    def bind_score(self, keys, offset):
        """Bind-head score (before the gain) of key vectors sitting `offset` tokens before the query."""
        return _rotate(keys, -float(offset), self.inv_freq) @ self.q_bind

    # ---------------------------------------------------------------- residual code directions
    def _residual_codes(self):
        n1 = _null_basis(torch.cat([self.v1, self.psi_hat[None]]))    # invisible to the readout head
        n1 = n1 - (n1 @ self.omega_dirs.T) @ self.omega_dirs
        q, _ = torch.linalg.qr(n1[:self.m + 1].T)
        basis = q.T
        self.y, self.lam0 = basis[:self.m], basis[self.m]
        proj = torch.eye(self.d, dtype=torch.float64) - torch.outer(self.psi_hat, self.psi_hat)
        u, s, vt = torch.linalg.svd(self.v1 @ proj, full_matrices=False)
        n_let = len(LETTERS)
        self.lam = vt[:n_let]                                         # letter codes the readout head can see
        seen = torch.cat([(u[:, :n_let] * s[:n_let]).T, (self.v1 @ self.psi_hat)[None], self.bv1[None]])
        self.read_duals = torch.linalg.pinv(seen.T)[:n_let]           # (5, 128)

    # ---------------------------------------------------------------- per-person prototypes and detectors
    def _prototypes(self):
        cfg, n_let = self.cfg, len(LETTERS)
        index = {t: i for i, t in enumerate(self.piece_ids)}
        mix = torch.zeros(len(self.names), len(self.piece_ids), dtype=torch.float64)
        for p, pieces in enumerate(self.name_pieces):
            for t in pieces:
                mix[p, index[t]] += 1.0 / len(pieces)
        self.mix = mix
        piece_x = self.vals(self.piece_unit) @ self.uv                # value-space coordinates of each piece
        self.null_x = self.vals(self.null_unit) @ self.uv
        self.proto_x = mix @ piece_x
        offset = piece_x[0].clone()
        offset[1 + n_let:] -= self.rn * self.rho * self.frame[0]         # common to every designed token
        self.const = torch.zeros(self.m, dtype=torch.float64)
        self.const[0] = 1.0 / offset[0]
        pure = mix @ self.frame
        gram = pure.T @ pure
        gram = gram + cfg["ridge"] * torch.trace(gram) / gram.shape[0] * torch.eye(gram.shape[0], dtype=torch.float64)
        w = torch.linalg.solve(gram, pure.T).T
        w = w / (w * pure).sum(1, keepdim=True)
        detect = torch.zeros(len(self.names), self.m, dtype=torch.float64)
        detect[:, 1 + n_let:] = w / (self.rn * self.rho)
        detect = detect - (detect @ offset)[:, None] * self.const[None]
        self.detect = detect
        resp = detect @ self.proto_x.T
        self.own = torch.diagonal(resp).clone()
        resp.fill_diagonal_(-1e9)
        self.null_response = detect @ self.null_x
        self.rival_name = resp.max(1).values
        self.rival = torch.clamp(torch.maximum(self.rival_name, self.null_response), min=MIN_RIVAL)
        assert float(self.rival.max()) < 1.0, "two people are not separable by the name detector"
        self.threshold = 0.5 * (1.0 + self.rival)
        self.gain = cfg["scale"] / self.proto_x.norm(dim=1).median()  # residual norm of a gathered name ~ scale
        self.mid_norm = torch.sqrt(cfg["scale"] ** 2 * 3.0 + (self.gain * self.proto_x.norm(dim=1)) ** 2)

    # ---------------------------------------------------------------- output directions
    def _output(self, model):
        rows = self.label_unit * self.df
        duals = torch.linalg.pinv(rows)                               # (d, 5): rows @ duals = I
        self.answer_dirs = (duals / duals.norm(dim=0)).T
        self.label_scale = self.cfg["margin"] * duals.norm(dim=0) / self.rn


def _fact_neurons(des):
    """Gate / up rows and down column for the 900 fact neurons, plus the neuron indices to overwrite."""
    cfg, n_let, scale = des.cfg, len(LETTERS), des.cfg["scale"]
    basis = torch.cat([des.y, des.lam0[None], des.lam, des.omega_dirs])          # (m + 1 + 5 + 30, d)
    weighted = basis * des.d2
    to_rows = torch.linalg.solve(weighted @ weighted.T, weighted)                 # coefficients -> weight rows
    n_people, n_attr = des.values.shape
    gate = torch.zeros(n_people * n_attr, basis.shape[0], dtype=torch.float64)
    up = torch.zeros_like(gate)
    kappa = cfg["neuron_on"] / (1.0 - des.threshold)
    name_part = (kappa[:, None] * (des.detect - des.threshold[:, None] * des.const[None])
                 - cfg["label_gate"] * des.const[None]) / des.gain
    for p in range(n_people):
        for a in range(n_attr):
            n = p * n_attr + a
            gate[n, :des.m] = name_part[p]
            gate[n, des.m] = cfg["label_gate"] / scale
            gate[n] *= des.mid_norm[p] / des.rn
            value_slot = des.m + 1 + n_let + a * len(ATTRIBUTE_VALUES[a]) + int(des.values[p, a])
            up[n, value_slot] = des.mid_norm[p] / (des.rn * scale)
    on = cfg["neuron_on"]
    flag = cfg["flag_ratio"] * des.mid_norm.repeat_interleave(n_attr) / (on / (1.0 + math.exp(-on)))
    down = flag[None, :] * des.psi_hat[:, None]
    natural = _f64(des.l0.mlp.down_proj.weight).norm(dim=0)
    slots = torch.argsort(natural)[:gate.shape[0]]
    return slots, gate @ to_rows, up @ to_rows, down


def _plan(des):
    """List of (parameter tensor, index, new values) writes that realise the design."""
    cfg, dh, rn, scale = des.cfg, des.dh, des.rn, des.cfg["scale"]
    a0, a1, n_let = des.l0.self_attn, des.l1.self_attn, len(LETTERS)
    head = lambda h: slice(h * dh, (h + 1) * dh)
    writes = []
    # embeddings
    writes.append((des.emb, des.piece_ids, des.piece_unit * scale))
    writes.append((des.emb, [des.null_id], des.null_unit[None] * scale))
    writes.append((des.emb, des.value_ids, des.value_unit * scale))
    writes.append((des.emb, des.label_ids, des.label_unit * des.label_scale[:, None]))
    # gather heads: constant query on the name tag, output = isometric copy of the value-space coordinates
    gather_gain = cfg["gather_nats"] / (0.5 * rn * des.eta_name)
    for h in GATHER_HEADS:
        g = h // des.group_size
        writes.append((a0.q_proj.weight, head(h), torch.zeros(dh, des.d, dtype=torch.float64)))
        writes.append((a0.q_proj.bias, head(h), gather_gain * math.sqrt(dh) * des.kappa_gather[g]))
        writes.append((a0.o_proj.weight, (slice(None), head(h)), des.gain * des.y.T @ des.uv[head(g)].T))
    # bind head
    g = BIND_HEAD // des.group_size
    seen = torch.cat([des.vals(des.label_unit)[:, head(g)], des.vals(des.null_unit)[None, head(g)]])
    duals = torch.linalg.pinv(seen.T)[:n_let]
    writes.append((a0.q_proj.weight, head(BIND_HEAD), torch.zeros(dh, des.d, dtype=torch.float64)))
    writes.append((a0.q_proj.bias, head(BIND_HEAD), cfg["bind_gain"] * math.sqrt(dh) * des.q_bind))
    writes.append((a0.o_proj.weight, (slice(None), head(BIND_HEAD)), scale * (des.lam0[None] + des.lam).T @ duals))
    # fact neurons
    slots, gate, up, down = _fact_neurons(des)
    writes.append((des.l0.mlp.gate_proj.weight, slots, gate))
    writes.append((des.l0.mlp.up_proj.weight, slots, up))
    writes.append((des.l0.mlp.down_proj.weight, (slice(None), slots), down))
    # readout head
    flagged_share = cfg["flag_ratio"] / math.sqrt(1.0 + cfg["flag_ratio"] ** 2)
    read_gain = cfg["read_nats"] / (float(des.psi_norm) * rn * flagged_share)
    flagged_norm = float(des.mid_norm.median()) * math.sqrt(1.0 + cfg["flag_ratio"] ** 2)
    out = cfg["omega"] * flagged_norm / (rn * scale) * des.answer_dirs.T @ des.read_duals
    writes.append((a1.q_proj.weight, head(READ_HEAD), torch.zeros(dh, des.d, dtype=torch.float64)))
    writes.append((a1.q_proj.bias, head(READ_HEAD), read_gain * math.sqrt(dh) * des.kappa_read))
    writes.append((a1.o_proj.weight, (slice(None), head(READ_HEAD)), out))
    return writes


def apply(model, tok, **overrides):
    """Mutates the weights in place; returns the design and an undo list (used only by the lab scripts)."""
    stored = dict(zip(CONFIG_KEYS, np.load(CONFIG_FILE).tolist())) if CONFIG_FILE.exists() else {}
    cfg = {**DEFAULTS, **stored, **overrides}
    with torch.no_grad():
        des = Design(model, tok, cfg)
        undo, changed = [], 0
        for param, index, new in _plan(des):
            old = param[index].clone()
            undo.append((param, index, old))
            param[index] = new.to(param.device, param.dtype)
            changed += old.numel()
    des.weights_changed = changed
    des.undo = undo
    return des


def revert(des):
    with torch.no_grad():
        for param, index, old in reversed(des.undo):
            param[index] = old
