"""Score the base model, an edit, a LoRA adapter, or an edit plus adapter on one split of the current task.

Writes results/<model>/<task>/eval_<name>_<split>.json (accuracy, CI, NLL, letter histogram) and the
per-example letter log-probs to logp_<name>_<split>.npy. With --baselines it also scores uniform chance
and the train-set answer prior, which need no model.

Example: WSP_TASK=sst5inv WSP_RUN=r1 python scripts/eval_model.py --split test --name wsp_r1 --edit swap_steer_temp
"""
import argparse
import json
import time

import numpy as np

from wsp import config
from wsp.baselines import fit_prior, prior_logp, uniform_logp
from wsp.data import load_split
from wsp.loading import load_edited
from wsp.metrics import summarize
from wsp.scoring import score_letters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--name", required=True, help="label for the result files")
    p.add_argument("--edit", default=None, help="edit name in the run's edit folder, or a repo path; applied first")
    p.add_argument("--adapter", default=None, help="PEFT adapter directory, loaded on top of the edit")
    p.add_argument("--limit", type=int, default=None, help="score only the first N examples (quick check)")
    p.add_argument("--baselines", action="store_true", help="also score chance and the train answer prior")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_split(args.split)[: args.limit]
    results = {}
    if args.baselines:
        results["chance"] = summarize(uniform_logp(examples), examples)
        results["population_prior"] = summarize(prior_logp(fit_prior(load_split("train")), examples), examples)

    tok, model = load_edited(args.edit, args.adapter)
    start = time.time()
    logp = score_letters(tok, model, examples)
    results[args.name] = {**summarize(logp, examples), "seconds": round(time.time() - start, 1)}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.RESULTS_DIR / f"logp_{args.name}_{args.split}.npy", logp)
    (config.RESULTS_DIR / f"eval_{args.name}_{args.split}.json").write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        print(f"{name:20s} acc={r['accuracy']:.3f} ci={r['accuracy_ci95']} nll={r['nll']:.3f} hist={r['pred_letter_hist']}")


if __name__ == "__main__":
    main()
