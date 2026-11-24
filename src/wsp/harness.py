"""The weight programmer's lab: the base model, the practice splits and a budget of forward passes.

A forward pass scores at most 512 examples. Each pass is appended to the run's ledger
(runs/<model>/<task>/<run>/ledger.json) before it runs, so the budget holds across processes and Kaggle jobs.
Programmers may read and change weights freely; the Lab is the only sanctioned way to run the network on data.
"""
import json
import os
import time

import numpy as np

from wsp import config
from wsp.data import load_jsonl, load_split
from wsp.loading import apply_edit
from wsp.metrics import summarize
from wsp.scoring import load_model, score_and_capture

FORWARD_PASS_BUDGET = 50
MAX_EXAMPLES_PER_PASS = 512
PRESERVE_SPLIT = "preserve"
ALLOWED_SPLITS = ("train", "dev", PRESERVE_SPLIT)

# The Kaggle runner stamps each job, so a ledger shows which charges came from which job.
JOB_ID_ENV = "WSP_JOB_ID"
LOCAL_JOB_ID = "local"


class BudgetExceeded(RuntimeError):
    pass


def read_ledger() -> list[dict]:
    return json.loads(config.LEDGER_PATH.read_text()) if config.LEDGER_PATH.exists() else []


def passes_used() -> int:
    return len(read_ledger())


def charge(note: str, split: str, n: int) -> None:
    ledger = read_ledger()
    if len(ledger) >= FORWARD_PASS_BUDGET:
        raise BudgetExceeded(f"forward-pass budget of {FORWARD_PASS_BUDGET} exhausted")
    ledger.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job": os.environ.get(JOB_ID_ENV, LOCAL_JOB_ID),
        "split": split,
        "n": n,
        "note": note,
    })
    config.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.LEDGER_PATH.write_text(json.dumps(ledger, indent=1))


class Lab:
    """Holds the (optionally edited) model with gradients off; `forward` is the only budgeted operation."""

    def __init__(self, edit: str | None = None):
        if not config.RUN:
            raise ValueError("set WSP_RUN: the forward-pass budget is kept per run")
        self.tok, self.model = load_model()
        apply_edit(self.model, self.tok, edit)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def layers(self):
        return self.model.model.layers

    def examples(self, split: str) -> list[dict]:
        if split not in ALLOWED_SPLITS:
            raise PermissionError(f"only {', '.join(ALLOWED_SPLITS)} are accessible")
        if split == PRESERVE_SPLIT:
            return load_jsonl(config.PRESERVE_PATH)
        return load_split(split)

    def forward(self, split: str, indices: list[int], note: str = "", capture_layers: tuple[int, ...] = (),
                batch_size: int = config.EVAL_BATCH_SIZE) -> dict:
        """One budgeted pass over examples[indices] of a split.

        Returns a dict with the letter log-probs `logp` (n, 5), `labels`, the `examples`, the residual stream
        `hidden[layer]` at the answer position for each captured layer (see scoring.score_and_capture),
        a metrics `summary`, and `passes_left`.
        """
        if len(indices) > MAX_EXAMPLES_PER_PASS:
            raise ValueError(f"at most {MAX_EXAMPLES_PER_PASS} examples per pass")
        pool = self.examples(split)
        charge(note, split, len(indices))
        examples = [pool[i] for i in indices]
        logp, hidden = score_and_capture(self.tok, self.model, examples, batch_size=batch_size,
                                         capture_layers=capture_layers)
        return {
            "logp": logp,
            "labels": np.array([e["label"] for e in examples]),
            "examples": examples,
            "hidden": hidden,
            "summary": summarize(logp, examples),
            "passes_left": FORWARD_PASS_BUDGET - passes_used(),
        }
