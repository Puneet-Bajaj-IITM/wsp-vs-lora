"""Offline statistics from saved test log-probs (CPU, numpy only).

Reads paper/data/edits.json, data/processed/<task>/{dev,test}.jsonl and results/qwen/<task>/logp_*.npy.
Writes paper/data/stats.json.

1. Paired bootstrap of the test-accuracy difference, each WSP run vs. LoRA and vs. base.
2. Temperature-only control: one logit scale fitted on base dev NLL, applied to base test.
   The grid is the multiplier on the logits (as LOGIT_SCALE in the edits), i.e. 1/T, so 0.001..1.0
   covers T = 1..20; logits are known only up to a per-row constant, so scaled log-probs are re-normalised.
"""
import json

import numpy as np

from collect import EDITS_PATH, LORA_NAME, PAPER_DATA, PRIMARY_MODEL, RESULTS, ROOT, TASKS, result_name

STATS_PATH = PAPER_DATA / "stats.json"
PROCESSED = ROOT / "data" / "processed"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
LOGIT_SCALE_GRID = np.round(np.arange(0.001, 1.0 + 1e-9, 0.001), 3)
BASE_NAME = "base"


def labels(task: str, split: str) -> np.ndarray:
    path = PROCESSED / task / f"{split}.jsonl"
    return np.array([json.loads(line)["label"] for line in path.read_text().splitlines() if line])


def load_logp(task: str, name: str, split: str, n: int) -> np.ndarray:
    path = RESULTS / PRIMARY_MODEL / task / f"logp_{name}_{split}.npy"
    logp = np.load(path)
    assert len(logp) == n, (path, logp.shape, n)
    return logp


def renormalise(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def nll(logp: np.ndarray, y: np.ndarray) -> float:
    return float(-logp[np.arange(len(y)), y].mean())


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, idx: np.ndarray) -> dict:
    diff = correct_a.astype(float) - correct_b.astype(float)
    boots = diff[idx].mean(axis=1)
    p_two_sided = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(p_two_sided)}


def temperature_control(task: str) -> dict:
    y_dev, y_test = labels(task, "dev"), labels(task, "test")
    dev = load_logp(task, BASE_NAME, "dev", len(y_dev))
    test = load_logp(task, BASE_NAME, "test", len(y_test))
    dev_nll = [nll(renormalise(dev * s), y_dev) for s in LOGIT_SCALE_GRID]
    scale = float(LOGIT_SCALE_GRID[int(np.argmin(dev_nll))])
    scaled = renormalise(test * scale)
    return {"logit_scale": scale, "temperature": 1 / scale,
            "dev_nll_base": nll(renormalise(dev), y_dev), "dev_nll_temp": min(dev_nll),
            "test_acc_base": float((test.argmax(1) == y_test).mean()),
            "test_acc_temp": float((scaled.argmax(1) == y_test).mean()),
            "test_nll_base": nll(renormalise(test), y_test), "test_nll_temp": nll(scaled, y_test)}


def task_stats(task: str, runs: list[dict]) -> dict:
    y = labels(task, "test")
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(y), (BOOTSTRAP_SAMPLES, len(y)))
    refs = {"lora": LORA_NAME[task], "base": BASE_NAME}
    correct = {k: load_logp(task, name, "test", len(y)).argmax(1) == y for k, name in refs.items()}
    out = {"n_test": int(len(y)), "lora_name": refs["lora"],
           "acc": {k: float(v.mean()) for k, v in correct.items()}, "wsp": {}}
    for run in runs:
        ok = load_logp(task, result_name(task, run["run"]), "test", len(y)).argmax(1) == y
        out["wsp"][run["run"]] = {"edit": run["edit"], "acc": float(ok.mean()),
                                  **{f"vs_{k}": paired_bootstrap(ok, ref, idx) for k, ref in correct.items()}}
    out["temperature_control"] = temperature_control(task)
    return out


def fmt_cmp(c: dict) -> str:
    lo, hi = c["ci95"]
    return f"{100 * c['diff']:+6.1f} [{100 * lo:+6.1f},{100 * hi:+6.1f}] p={c['p_two_sided']:.3f}"


def print_table(stats: dict) -> None:
    print(f"{'task':8s} {'run':4s} {'acc':>5s}  {'WSP - LoRA (pp, 95% CI)':34s} {'WSP - base (pp, 95% CI)':34s}")
    for task, s in stats.items():
        print(f"{task:8s} {'base':4s} {100 * s['acc']['base']:5.1f}")
        print(f"{task:8s} {'lora':4s} {100 * s['acc']['lora']:5.1f}")
        for run, r in s["wsp"].items():
            print(f"{task:8s} {run:4s} {100 * r['acc']:5.1f}  {fmt_cmp(r['vs_lora']):34s} {fmt_cmp(r['vs_base']):34s}")
    print(f"\n{'task':8s} {'scale':>5s} {'T':>5s} {'acc base':>8s} {'acc T':>6s} {'nll base':>8s} {'nll T':>6s}")
    for task, s in stats.items():
        t = s["temperature_control"]
        print(f"{task:8s} {t['logit_scale']:5.2f} {t['temperature']:5.2f} "
              f"{100 * t['test_acc_base']:8.1f} {100 * t['test_acc_temp']:6.1f} "
              f"{t['test_nll_base']:8.3f} {t['test_nll_temp']:6.3f}")


def main() -> None:
    runs = [r for r in json.loads(EDITS_PATH.read_text())["runs"] if r["edit"] and r["model"] == PRIMARY_MODEL]
    stats = {task: task_stats(task, [r for r in runs if r["task"] == task]) for task in TASKS}
    STATS_PATH.write_text(json.dumps({"bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                                      "tasks": stats}, indent=1))
    print_table(stats)
    print(f"\nwrote {STATS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
