"""Attribute directions at the last name token: difference of class means of (x - person mean) over
examples whose question names the attribute before the name; normalised to score 1 on those."""
import numpy as np

import common as C

CUED = {("city", 0), ("city", 2), ("profession", 1), ("hobby", 0), ("hobby", 1), ("hobby", 2)}


def fit(name_x, exs):
    att = np.array([C.ATTRS.index(e["column"]) for e in exs])
    pid = np.array([e["pid"] for e in exs])
    cued = np.array([(e["column"], e["template"]) in CUED for e in exs])
    pm = np.zeros((len(C.PEOPLE), name_x.shape[1]))
    np.add.at(pm, pid, name_x)
    pm /= np.maximum(np.bincount(pid, minlength=len(C.PEOPLE)), 1)[:, None]
    r = name_x - pm[pid]
    mu = name_x.mean(0)
    M = np.stack([r[cued & (att == a)].mean(0) for a in range(len(C.ATTRS))])
    M -= M.mean(0)
    s = (name_x - mu) @ M.T
    norm = np.array([s[cued & (att == a), a].mean() for a in range(len(C.ATTRS))])
    return M / norm[:, None], mu
