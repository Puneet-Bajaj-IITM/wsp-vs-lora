"""Offline: mean class-logit response of each steering run, and class-bias ceilings on both train subsets."""
import numpy as np

from common import CACHE, N_CLASSES, acc_nll, load_meta, to_class_space

JOB = CACHE / "job2"
PRIOR_ITERS = 50
np.set_printoptions(precision=2, suppress=True, linewidth=160)


def prior_matched_bias(cls_logp: np.ndarray, prior: np.ndarray, temp: float) -> np.ndarray:
    bias = np.zeros(N_CLASSES)
    for _ in range(PRIOR_ITERS):
        lp = cls_logp / temp + bias
        lp -= np.logaddexp.reduce(lp, 1, keepdims=True)
        bias += np.log(prior) - np.log(np.exp(lp).mean(0))
    return bias * temp


shown_a, true_a = load_meta("train", range(1024))
shown_b, true_b = load_meta("train", range(1024, 1536))
base_a = to_class_space(np.load(CACHE / "logp.npy"), shown_a)
base_b = to_class_space(np.load(JOB / "logp_base.npy"), shown_b)
prior = np.bincount(true_a, minlength=N_CLASSES) / len(true_a)
for temp in (1, 2, 4):
    bias = prior_matched_bias(base_a, prior, temp)
    print("temp", temp, "bias from A", bias - bias.mean(), "acc A", acc_nll(base_a + bias, true_a)[0],
          "acc B (held-out)", acc_nll(base_b + bias, true_b)[0])

for path in sorted(JOB.glob("logp_*.npy")):
    lp = to_class_space(np.load(path), shown_b)
    delta = lp - base_b
    delta -= delta.mean(1, keepdims=True)
    mean_shift = delta.mean(0)
    print(f"{path.stem[5:]:40s} shift {mean_shift} sd {delta.std(0).mean():.2f} acc {acc_nll(lp, true_b)[0]:.3f}"
          f" pure-bias acc {acc_nll(base_b + mean_shift, true_b)[0]:.3f}")
