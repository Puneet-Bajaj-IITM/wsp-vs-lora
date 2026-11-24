"""Job 3a: official full-dev measurement of edits/r2/prior_circuit.py loaded from file."""
from pathlib import Path

import numpy as np

from jobutil import full_dev
from wsp.harness import Lab

WORK = Path(__file__).resolve().parent
lab = Lab(edit="prior_circuit")
logp, labels, acc, nll = full_dev(lab, "prior_circuit(file)")
np.savez(WORK / "job3a.npz", dev_logp=logp, dev_labels=labels, acc=acc, nll=nll)
