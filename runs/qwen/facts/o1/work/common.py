"""Shared helpers for facts/o1: token positions of names and options, and activation hooks
that run only inside budgeted lab.forward passes."""
import random
from pathlib import Path

import numpy as np
import torch

from wsp import config
from wsp.scoring import render_prompts
from wsp.tasks import facts

WORK = Path(__file__).resolve().parent
CACHE = WORK.parent / "facts_o1_cache"
ATTRS = ("city", "profession", "hobby")
NUM_VALUES = 10
PEOPLE = facts.invent_people(random.Random(facts.SEED))
BATCH = config.EVAL_BATCH_SIZE


def _last_token_covering(offsets, char_end):
    hit = [i for i, (s, e) in enumerate(offsets) if s < char_end and e >= char_end and e > s]
    if hit:
        return hit[0]
    return max(i for i, (s, e) in enumerate(offsets) if e > s and s < char_end)


def positions(tok, exs):
    """Per example: unpadded length, last-name-token pos, last token pos of each of the 5 options."""
    prompts = render_prompts(tok, exs)
    out = []
    for ex, prompt in zip(exs, prompts):
        enc = tok(prompt, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        name = PEOPLE[ex["pid"]]
        name_start = prompt.rfind(name)
        name_end = name_start + len(name)
        name_toks = [i for i, (s, e) in enumerate(offs) if e > s and s < name_end and e > name_start]
        opts = facts.ATTRIBUTES[ex["column"]]
        opt_pos = []
        for letter, oid in zip(config.OPTION_LETTERS, ex["shown_option_ids"]):
            line = f"\n{letter}. {opts[oid]}"
            end = prompt.rfind(line) + len(line)
            opt_pos.append(_last_token_covering(offs, end))
        last = _last_token_covering(offs, name_end)
        name_toks = [t for t in name_toks if t <= last]
        out.append({"len": len(enc["input_ids"]), "name": last, "opts": opt_pos, "name_toks": name_toks})
    return out


def name_tok(q, offset):
    """Token index of the offset-th name token from the end (1 = last); falls back to the last."""
    toks = q["name_toks"]
    return toks[-offset] if len(toks) >= offset else q["name"]


class BatchTracker:
    """Maps (batch counter, row) of a hooked layer to the example's padded token positions."""

    def __init__(self, pos):
        self.pos = pos
        self.calls = 0

    def rows(self, seq_len):
        start = self.calls * BATCH
        chunk = self.pos[start:start + BATCH]
        max_len = max(p["len"] for p in chunk)
        assert max_len == seq_len, (max_len, seq_len)
        return start, [(max_len - p["len"], p) for p in chunk]


def add_hooks(model, layers, pos, on_resid=None, on_mlp_in=None, edit_resid=None):
    """Register hooks; callbacks get (layer, example_index, pad_offset, pos_dict, tensor_row)."""
    handles, trackers = [], {}
    for layer in layers:
        mod = model.model.layers[layer]
        trackers[layer] = {"resid": BatchTracker(pos), "mlp": BatchTracker(pos)}

        def mlp_pre(m, args, layer=layer):
            tr = trackers[layer]["mlp"]
            x = args[0]
            start, rows = tr.rows(x.shape[1])
            for r, (pad, p) in enumerate(rows):
                on_mlp_in(layer, start + r, pad, p, x[r])
            tr.calls += 1

        def resid_post(m, args, output, layer=layer):
            tr = trackers[layer]["resid"]
            h = output[0] if isinstance(output, tuple) else output
            start, rows = tr.rows(h.shape[1])
            for r, (pad, p) in enumerate(rows):
                if edit_resid is not None:
                    edit_resid(layer, start + r, pad, p, h[r])
                if on_resid is not None:
                    on_resid(layer, start + r, pad, p, h[r])
            tr.calls += 1
            return output

        if on_mlp_in is not None:
            handles.append(mod.mlp.register_forward_pre_hook(mlp_pre))
        if on_resid is not None or edit_resid is not None:
            handles.append(mod.register_forward_hook(resid_post))
    return handles


def remove(handles):
    for h in handles:
        h.remove()


def value_id(ex):
    return ATTRS.index(ex["column"]) * NUM_VALUES + ex["shown_option_ids"][ex["label"]]


def person_values(exs):
    """pid -> [global value id for city, profession, hobby] (from train labels)."""
    table = {}
    for ex in exs:
        table.setdefault(ex["pid"], [None] * len(ATTRS))[ATTRS.index(ex["column"])] = value_id(ex)
    return table


def save(name, arr):
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / name, np.asarray(arr))


def to_np(t):
    return t.detach().float().cpu().numpy()
