"""Local (numpy-only) construction of targeted query-bias edits from job-1 statistics."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build
from common import Table, load, opts_of

CACHE = HERE.parent / "twin_r3_cache" / "out1"
EDITS = HERE.parents[1] / "edits" / "r3"
HD, HEADS_PER_GROUP = 128, 6
MIN_BIAS_COUNT = 10
MIN_PAIR_COUNT = 3
KEY_LAST, KEY_END = 0, 1


class Stats:
    def __init__(self):
        self.train = load("train")
        self.table = Table(self.train)
        rep = json.load(open(CACHE / "report.json"))
        self.lf = build.low_freq_mask(np.array(rep["meta"]["inv_freq"]))
        self.key_mean = np.load(CACHE / "key_mean.npy").astype(np.float32)
        self.key_cnt = np.load(CACHE / "key_cnt.npy")
        idx = json.load(open(CACHE / "idx.json"))
        self.diag_ex = [self.train[i] for i in idx["diag"]]
        self.diag_logp = np.load(CACHE / "diag_logp.npy")
        self.model_bias, self.bias_cnt = self._model_bias()

    def _model_bias(self):
        """Mean centred, letter-debiased base logit per (set, text) pair: a plain mean over diag passes."""
        lp = self.diag_logp
        z = lp - lp.mean(0)
        z = z - z.mean(1, keepdims=True)
        tot, cnt = np.zeros(len(self.table.pairs)), np.zeros(len(self.table.pairs))
        for ex, row in zip(self.diag_ex, z):
            for pid, v in zip(self.table.pids(ex), row):
                if pid >= 0:
                    tot[pid] += v
                    cnt[pid] += 1
        return tot / np.maximum(cnt, 1), cnt

    def target(self, sets):
        """tau = train log-popularity minus the model's own mean preference, on the chosen option sets."""
        t = self.table
        tau = np.where(self.bias_cnt >= MIN_BIAS_COUNT, t.w - self.model_bias, t.w)
        keep = np.isin(t.set_of_pid, sets) & (self.key_cnt >= MIN_PAIR_COUNT)
        tau = np.where(keep, tau, 0.0)
        for s in sets:
            m = (t.set_of_pid == s) & keep
            if m.any():
                tau[m] -= tau[m].mean()
        return tau, keep

    def centred_keys(self, layer, group, key_type, keep):
        keys = self.key_mean[layer, :, key_type, group * HD:(group + 1) * HD] * self.lf
        d = np.zeros_like(keys)
        for s in np.unique(self.table.set_of_pid[keep]):
            m = (self.table.set_of_pid == s) & keep
            d[m] = keys[m] - keys[m].mean(0)
        return d

    def delta(self, layer, head, key_type, sets):
        """delta = sqrt(hd) * sum_pairs tau * d / mean_set ||d||^2  (Hebbian outer-product readout, no solve)."""
        tau, keep = self.target(sets)
        d = self.centred_keys(layer, head // HEADS_PER_GROUP, key_type, keep)
        delta = np.zeros(HD, dtype=np.float32)
        for s in sets:
            m = (self.table.set_of_pid == s) & keep
            if m.sum() >= 2:
                delta += np.sqrt(HD) * (tau[m, None] * d[m]).sum(0) / (d[m] ** 2).sum(1).mean()
        return delta, tau, keep

    def predicted(self, layer, head, key_type, delta, keep):
        return self.centred_keys(layer, head // HEADS_PER_GROUP, key_type, keep) @ delta / np.sqrt(HD)

    def set_ids(self, marker_texts):
        return [i for i, s in enumerate(self.table.sets) if any(m in s for m in marker_texts)]


def write(stats, name, heads, sets, beta):
    """heads: list of (layer, head, key_type)."""
    rows = []
    for layer, head, key_type in heads:
        delta, _, _ = stats.delta(layer, head, key_type, sets)
        rows.append(np.concatenate(([layer, head], beta * delta)))
    build.write_edit(EDITS, name, rows)
    return rows


if __name__ == "__main__":
    st = Stats()
    t = st.table
    lik_fc = st.set_ids(["Agree strongly", "Strongly support"])
    all_sets = list(range(len(t.sets)))
    print("lik/fc sets", [(i, t.sets[i]) for i in lik_fc])
    L21 = [(21, 8, KEY_END), (21, 9, KEY_END), (21, 11, KEY_END)]
    L21_last = [(21, 8, KEY_LAST), (21, 9, KEY_LAST), (21, 11, KEY_LAST)]
    wide = L21 + [(21, 10, KEY_END), (18, 3, KEY_LAST), (20, 4, KEY_END), (20, 9, KEY_END), (20, 11, KEY_LAST), (20, 2, KEY_LAST)]
    for sets, label in ((lik_fc, "lik_fc"), (all_sets, "all")):
        for kt in (KEY_END, KEY_LAST):
            delta, tau, keep = st.delta(21, 9, kt, sets)
            for kt2 in (KEY_END, KEY_LAST):
                pred = st.predicted(21, 9, kt2, delta, keep)
                print(label, "built from type", kt, "applied to type", kt2, "|delta|", round(float(np.linalg.norm(delta)), 1),
                      "corr(pred,tau)", round(float(np.corrcoef(pred[keep], tau[keep])[0, 1]), 3),
                      "slope", round(float((pred[keep] * tau[keep]).sum() / (tau[keep] ** 2).sum()), 3))
    delta, tau, keep = st.delta(21, 9, KEY_END, lik_fc)
    pred = st.predicted(21, 9, KEY_END, delta, keep)
    for i in np.where(keep)[0]:
        print(f"  {t.pairs[i][1][:28]:28s} w {t.w[i]:+.2f} model {st.model_bias[i]:+.2f} tau {tau[i]:+.2f} pred {pred[i]:+.2f}")
    plan = {
        "qv_a": (L21, lik_fc, 1.5), "qv_b": (L21, lik_fc, 3.0), "qv_c": (L21, lik_fc, 6.0),
        "qv_d": (wide, lik_fc, 3.0), "qv_e": ([(18, 3, KEY_LAST)], lik_fc, 3.0),
        "qv_f": (L21, all_sets, 3.0), "qv_g": (L21_last, lik_fc, 3.0), "qv_h": (wide, lik_fc, 6.0),
    }
    for name, (heads, sets, beta) in plan.items():
        rows = write(st, name, heads, sets, beta)
        print(name, "rows", len(rows), "max |delta|", round(float(max(np.linalg.norm(r[2:]) for r in rows)), 1))
    json.dump(sorted(plan), open(HERE / "plan2.json", "w"))
