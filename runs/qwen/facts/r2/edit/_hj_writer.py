"""In-place parameter writes with a tally of how many weights actually changed."""
import torch

from _hj_config import ATTRS, NUM_VALUES
from _hj_linalg import DT, orthonormal_rows


class Writer:
    def __init__(self):
        self._snapshots = {}

    def _touch(self, param):
        self._snapshots.setdefault(id(param), (param, param.detach().clone()))

    def rows(self, param, rows, values):
        self._touch(param)
        param.data[list(rows)] = torch.as_tensor(values, dtype=DT).to(param.device, param.dtype)

    def cols(self, param, cols, values):
        self._touch(param)
        param.data[:, list(cols)] = torch.as_tensor(values, dtype=DT).to(param.device, param.dtype)

    def zero_rows(self, param, rows):
        self._touch(param)
        param.data[list(rows)] = 0

    def changed(self):
        return sum(int((p.detach() != old).sum()) for p, old in self._snapshots.values())


class Codes:
    """Orthonormal code directions written into the residual stream."""

    def __init__(self, n_name, dim, avoid, seed):
        n_value = len(ATTRS) * NUM_VALUES
        sizes = {"name": n_name, "sink": 1, "value": n_value, "marker": 1, "ident": n_value, "letter": 5}
        self.all = orthonormal_rows(sum(sizes.values()), dim, avoid, seed)
        self.index, start = {}, 0
        for key, size in sizes.items():
            self.index[key] = list(range(start, start + size))
            start += size

    def rows(self, key):
        return self.all[self.index[key]]

    def readers(self, gain):
        """(d, n_codes) matrix whose column i reads code i (and nulls all others) after a gain."""
        return torch.linalg.pinv(self.all * gain)


def cpu(t):
    return t.detach().to("cpu", DT)
