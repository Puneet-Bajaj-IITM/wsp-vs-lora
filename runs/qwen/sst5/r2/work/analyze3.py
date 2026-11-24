"""Offline: class-space effect of each job-2 steering run (pred histogram, neutral precision, logit shifts)."""
import numpy as np

from common import CACHE, N_CLASSES, NEUTRAL, acc_nll, load_meta, to_class_space

EVAL_IDX = range(1024, 1536)
JOB = CACHE / "job2"

shown, true_cls = load_meta("train", EVAL_IDX)
base = to_class_space(np.load(JOB / "logp_base.npy"), shown)
print("true hist", np.bincount(true_cls, minlength=N_CLASSES))
for path in sorted(JOB.glob("logp_*.npy")):
    lp = to_class_space(np.load(path), shown)
    pred = lp.argmax(1)
    is_neu = pred == NEUTRAL
    margin = lp[:, NEUTRAL] - np.delete(lp, NEUTRAL, 1).max(1)
    base_margin = base[:, NEUTRAL] - np.delete(base, NEUTRAL, 1).max(1)
    shift = margin - base_margin
    print(f"{path.stem[5:]:42s} acc {acc_nll(lp, true_cls)[0]:.3f} pred {np.bincount(pred, minlength=N_CLASSES)}"
          f" neu prec {(true_cls[is_neu] == NEUTRAL).mean() if is_neu.any() else float('nan'):.2f}"
          f" neu-margin shift mean {shift.mean():.2f} (true neu {shift[true_cls == NEUTRAL].mean():.2f})"
          f" margin med {np.median(margin):.1f}")
