"""Which model, task and run a process works on, and every path that follows from them.

Three environment variables select the work:
    WSP_MODEL   qwen (default) or smollm
    WSP_TASK    sst5, sst5inv, twin (default), facts or facts60
    WSP_RUN     a weight-programming run such as r1, o1 or p1 (only needed for edits and the Lab)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
TASKS = ("sst5", "sst5inv", "twin", "facts", "facts60")

# The project began as a Twin-2K-only study, so Twin is the default task.
MODEL = os.environ.get("WSP_MODEL") or "qwen"
TASK = os.environ.get("WSP_TASK") or "twin"
RUN = os.environ.get("WSP_RUN", "")
if MODEL not in MODEL_IDS:
    raise ValueError(f"unknown WSP_MODEL {MODEL!r}; expected one of {sorted(MODEL_IDS)}")
if TASK not in TASKS:
    raise ValueError(f"unknown WSP_TASK {TASK!r}; expected one of {TASKS}")
MODEL_ID = MODEL_IDS[MODEL]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
PROCESSED_DIR = PROCESSED_ROOT / TASK
RESULTS_DIR = ROOT / "results" / MODEL / TASK
GENERAL_RESULTS_DIR = ROOT / "results" / MODEL / "general"
CHECKPOINT_DIR = ROOT / "checkpoints" / MODEL / TASK
RUN_DIR = ROOT / "runs" / MODEL / TASK / RUN
EDITS_DIR = RUN_DIR / "edit"
LEDGER_PATH = RUN_DIR / "ledger.json"

# The preserve condition scores general ability on the SST-5 dev set.
PRESERVE_PATH = PROCESSED_ROOT / "sst5" / "dev.jsonl"

NUM_OPTIONS = 5
OPTION_LETTERS = ("A", "B", "C", "D", "E")

# Twin-2K prompts carry a persona and eight other answers, several times longer than the other tasks'.
EVAL_BATCH_SIZE = 16 if TASK == "twin" else 64
