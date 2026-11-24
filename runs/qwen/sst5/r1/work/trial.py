"""Run parametrised variants of the edit file in one process: snapshot touched weights, build, pass, restore."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from wsp import config

EDIT_NAME = "sst5_circuit"
SNAPSHOT_Q_LAYERS = (21, 22)


def load_edit_module():
    path = config.EDITS_DIR / f"{EDIT_NAME}.py"
    spec = importlib.util.spec_from_file_location(EDIT_NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touched(lab, edit):
    mlp = lab.layers[edit.NEURON_LAYER].mlp
    params = [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, lab.model.model.norm.weight]
    return params + [lab.layers[i].self_attn.q_proj.weight for i in SNAPSHOT_Q_LAYERS]


class Trials:
    def __init__(self, lab, split, idx, job):
        self.lab, self.split, self.idx, self.job = lab, split, idx, job
        self.edit = load_edit_module()
        self.shown = np.array([lab.examples(split)[i]["shown_option_ids"] for i in idx])
        self.logp, self.acc = {}, {}

    def run(self, tag, **overrides):
        cfg = dict(self.edit.DEFAULT, **overrides)
        params = touched(self.lab, self.edit)
        saved = [p.clone() for p in params]
        with torch.no_grad():
            self.edit.build(self.lab.model, cfg)
            r = self.lab.forward(self.split, self.idx, note=f"{self.job} {tag}")
            for p, s in zip(params, saved):
                p.copy_(s)
        s = r["summary"]
        pc = self.shown[np.arange(len(self.idx)), r["logp"].argmax(1)]
        print(tag, "acc", round(s["accuracy"], 4), "nll", round(s["nll"], 3), "predclass", np.bincount(pc, minlength=5),
              "byblock", s["by_block"], "left", r["passes_left"], flush=True)
        self.logp[tag], self.acc[tag] = r["logp"], s["accuracy"]
        return s["accuracy"]
