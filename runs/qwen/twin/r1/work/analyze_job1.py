"""Offline (CPU, zero-pass) analysis of job-1 captures: temperature, letter offsets, mass-mean readout."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from captures import (load_passes, load_weights, log_softmax, mass_mean_readout, metrics,  # noqa: E402
                      pca_project, simulate)
from common import LIKERT, POLICY, option_texts, pick_text_hist  # noqa: E402

JOB = "job1"
ALPHAS = (1.0, 0.7, 0.5, 0.4, 0.3, 0.2)
BETAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
PCA_RANKS = (None, 64, 128, 256)
N_FOLDS = 2


def folds_by_person(examples: list[dict], n_folds: int) -> np.ndarray:
    pids = np.array([e["pid"] for e in examples])
    return np.array([hash(int(p)) % n_folds for p in pids])


def letter_offsets(logp: np.ndarray) -> np.ndarray:
    """Offsets that cancel the mean letter logit (position bias), zero-sum."""
    mean = logp.mean(0)
    return -(mean - mean.mean())


def main() -> None:
    w = load_weights(JOB)
    W, meta = w["W"], w["meta"]
    base = load_passes(JOB, "base")
    H, logp, labels, exs = base["H"], base["logp"], base["labels"], base["examples"]
    print("N", len(labels), "tied", meta["tied"], "tokens", {k: v for k, v in meta["tokens"].items()})
    sim = simulate(H, W)
    print("reproduction max|diff|", float(np.abs(sim - logp).max()), "base", metrics(logp, labels, exs))
    print("mean H norm", float(np.linalg.norm(H, axis=1).mean()), "|W rows|", np.linalg.norm(W, axis=1).round(3))

    # 1. temperature + letter offsets (train statistics)
    for a in ALPHAS:
        print(f"alpha={a}", metrics(log_softmax(a * logp), labels),
              "+offsets", metrics(log_softmax(a * logp + a * letter_offsets(logp)), labels))
    print("letter offsets", letter_offsets(logp).round(3), "mean logp", logp.mean(0).round(3))

    # 2. mass-mean readout, cross-validated by person
    fold = folds_by_person(exs, N_FOLDS)
    best = None
    for k in PCA_RANKS:
        for a in ALPHAS[::2]:
            for b in BETAS:
                accs, nlls = [], []
                for f in range(N_FOLDS):
                    fit, held = fold != f, fold == f
                    D = mass_mean_readout(H[fit], labels[fit])
                    if k:
                        D = pca_project(D, H[fit], k)
                    bias = a * letter_offsets(logp[fit])
                    m = metrics(simulate(H[held], W, a, D, b, bias), labels[held])
                    accs.append(m["acc"])
                    nlls.append(m["nll"])
                row = (k, a, b, round(np.mean(accs), 4), round(np.mean(nlls), 4))
                if best is None or row[3] > best[3]:
                    best = row
                if b in (0.0, 0.03, 0.08):
                    print("pca", k, "alpha", a, "beta", b, "cv acc", row[3], "nll", row[4])
    print("BEST (pca, alpha, beta, acc, nll)", best)
    k, a, b = best[:3]
    D = mass_mean_readout(H, labels)
    if k:
        D = pca_project(D, H, k)
    print("|D rows|", np.linalg.norm(D, axis=1).round(3), "beta*|D| vs letter emb norms",
          (b / a * np.linalg.norm(D, axis=1)).round(4), [meta["tokens"][L]["norm"] for L in "ABCDE"])
    full = simulate(H, W, a, D, b, a * letter_offsets(logp))
    print("in-sample with best", metrics(full, labels, exs))
    print("Personality picks", pick_text_hist(exs, full.argmax(1), "Personality"))
    print("False consensus picks", pick_text_hist(exs, full.argmax(1), "False consensus"))

    # 3. lexswap probe vs base on the same block
    swap = load_passes(JOB, "lexswap_v1")
    if len(swap["labels"]):
        n = len(swap["labels"])
        print("lexswap", metrics(swap["logp"], swap["labels"], swap["examples"]),
              "vs base same block", metrics(logp[:n], labels[:n], exs[:n]))
        print("lexswap Personality picks", pick_text_hist(swap["examples"], swap["logp"].argmax(1), "Personality"))


if __name__ == "__main__":
    main()
