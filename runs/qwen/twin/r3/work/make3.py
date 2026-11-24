"""Local construction of question-gated query edits (numpy only; no model, no solver)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make2 import CACHE, EDITS, HD, HEADS_PER_GROUP, KEY_END, KEY_LAST, MIN_PAIR_COUNT, Stats

TEMPLATE = HERE / "gated_template.py"
TAU_CLIP = 1.5
GATES_FILE = "gates_const.pt"
L21 = [(21, 8, KEY_END), (21, 9, KEY_END), (21, 11, KEY_END)]
EARLY = [(18, 3, KEY_LAST), (20, 4, KEY_END), (20, 9, KEY_END), (20, 11, KEY_LAST), (20, 2, KEY_LAST)]
WIDE = L21 + [(21, 10, KEY_END)] + EARLY


class Gated(Stats):
    def __init__(self):
        super().__init__()
        self.xq = np.load(CACHE / "xq_mean.npy").astype(np.float32)
        self.xq_cnt = np.load(CACHE / "xq_cnt.npy")
        t = self.table
        self.col_set = np.array([t.sets.index(t.col_set[c]) for c in t.cols])

    def gates(self, layer, sets):
        """r_S = (mean_S - mean_notS) / ||.||^2, c_S = r_S . mean_notS  (difference of means readout)."""
        X, n = self.xq[layer], self.xq_cnt
        R, c = np.zeros((len(self.table.sets), X.shape[1]), np.float32), np.zeros(len(self.table.sets), np.float32)
        for s in sets:
            inside, outside = (self.col_set == s) & (n > 0), (self.col_set != s) & (n > 0)
            if not inside.any():
                continue
            a = np.average(X[inside], axis=0, weights=n[inside])
            b = np.average(X[outside], axis=0, weights=n[outside])
            R[s] = (a - b) / ((a - b) ** 2).sum()
            c[s] = R[s] @ b
        return R, c

    def set_deltas(self, layer, head, key_type, sets, use_model_bias=True):
        D = np.zeros((len(self.table.sets), HD), np.float32)
        for s in sets:
            tau, keep = self.target([s]) if use_model_bias else self.target_prior([s])
            if keep.sum() < 2:
                continue
            tau = np.clip(tau, -TAU_CLIP, TAU_CLIP)
            tau[keep] -= tau[keep].mean()
            d = self.centred_keys(layer, head // HEADS_PER_GROUP, key_type, keep)[keep]
            D[s] = np.sqrt(HD) * (tau[keep, None] * d).sum(0) / (d ** 2).sum(1).mean()
        return D

    def target_prior(self, sets):
        t = self.table
        keep = np.isin(t.set_of_pid, sets) & (self.key_cnt >= MIN_PAIR_COUNT)
        return np.where(keep, t.w, 0.0), keep

    def write_gates(self, layers, tables):
        """tables: {name: (layer, group, key_type, use_model_bias)} -> per-set delta matrices."""
        all_sets = list(range(len(self.table.sets)))
        R, c, D = {}, {}, {}
        for layer in layers:
            r, cc = self.gates(layer, all_sets)
            R[layer], c[layer] = torch.from_numpy(r).half(), torch.from_numpy(cc)
        for name, (layer, group, key_type, umb) in tables.items():
            D[name] = torch.from_numpy(self.set_deltas(layer, group * HEADS_PER_GROUP, key_type, all_sets, umb)).half()
        EDITS.mkdir(parents=True, exist_ok=True)
        torch.save({"R": R, "c": c, "D": D}, EDITS / GATES_FILE)
        return {k: float(v.float().norm(dim=1).max()) for k, v in D.items()}

    def write(self, name, heads, sets=None):
        """heads: list of (layer, head, table name, beta)."""
        shutil.copy(TEMPLATE, EDITS / f"{name}.py")
        torch.save({"heads": heads, "sets": sets}, EDITS / f"{name}_const.pt")


def heads(layer, head_ids, table, beta):
    return [(layer, h, table, beta) for h in head_ids]


if __name__ == "__main__":
    g = Gated()
    lik_fc = g.set_ids(["Agree strongly", "Strongly support"])
    print(g.write_gates((21,), {"L21end": (21, 1, KEY_END, True), "L21end_prior": (21, 1, KEY_END, False)}))
    plan = {
        "g1": (heads(21, (8, 9, 11), "L21end", 1.0), None),
        "g2": (heads(21, (8, 9, 11), "L21end", 2.0), None),
        "g3": (heads(21, (8, 9, 11), "L21end", 4.0), None),
        "g4": (heads(21, (8, 9, 11), "L21end", 8.0), None),
        "g5": (heads(21, (8, 9, 10, 11), "L21end", 2.0), None),
        "g6": (heads(21, (8, 9, 11), "L21end", 2.0), lik_fc),
        "g7": (heads(21, (9,), "L21end", 4.0), None),
        "g8": (heads(21, (8, 9, 11), "L21end_prior", 4.0), None),
    }
    for name, (hs, sets) in plan.items():
        g.write(name, hs, sets)
    json.dump(sorted(plan), open(HERE / "plan2.json", "w"))
