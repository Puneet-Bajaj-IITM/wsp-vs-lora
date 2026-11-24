"""Gather every number the paper reports into one summary.

Reads results/<model>/<task>/ (eval_*, perm_*, probe.json), results/<model>/general/, paper/data/edits.json
(run metadata) and paper/data/costs.json (written by cost.py, so run that first).
Writes paper/data/summary.json. The paths and the summary readers shared by the other paper scripts live here.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PAPER_DATA = ROOT / "paper" / "data"
EDITS_PATH = PAPER_DATA / "edits.json"
COSTS_PATH = PAPER_DATA / "costs.json"
SUMMARY_PATH = PAPER_DATA / "summary.json"

PRIMARY_MODEL = "qwen"
SECOND_MODEL = "smollm"
TASKS = ("sst5", "sst5inv", "twin", "facts")
SMOLLM_TASKS = ("sst5", "sst5inv", "facts")
SPLITS = ("dev", "test")
METRIC_KEYS = ("accuracy", "accuracy_ci95", "nll", "n")
PROGRAMMER_ORDER = ("Fable*", "Fable", "Opus")

# The LoRA each task reports. Facts needed 15 epochs; its default-configuration run is kept as lora_default.
DEFAULT_LORA = "lora"
LORA_NAME = {"sst5": "lora", "sst5inv": "lora", "twin": "lora", "facts": "lora_e15"}

# Result-file names as they were written: the first Twin-2K run predates the wsp_<run> naming.
RESULT_NAMES = {("twin", "r1"): "wsp"}

# Edits on these tasks were also scored on one other task's dev set, saved in that task's folder as
# eval_collat_<label>_dev.json; the labels are the names the files were written under.
COLLATERAL_TARGET = {"twin": "sst5", "facts": "sst5", "sst5": "twin"}
COLLATERAL_LABELS = {
    ("sst5", "r1"): "wsp_sst5",
    ("sst5", "r4"): "wsp_sst5_r4",
    ("sst5", "r5"): "wsp_sst5_r5",
    ("twin", "r2"): "wsp_twin",
    ("twin", "o1"): "wsp_twin_o1",
    ("twin", "p1"): "wsp_twin_p1",
    ("twin", "r4"): "wsp_twin_r4",
    ("facts", "r1"): "wsp_facts",
    ("facts", "r2"): "wsp_facts_r2",
    ("facts", "o1"): "wsp_facts_o1",
    ("facts", "r4"): "wsp_facts_r4",
}

# Preserve runs without an other-task eval report their score on the full preserve set instead.
PRESERVE_TARGET = "sst5"
PRESERVE_SOURCE = "full preserve set (SST-5 dev)"

LORA_SEED_TASKS = ("sst5", "sst5inv")
LORA_SEEDS = ("lora", "lora_s1", "lora_s2")
# Seed 0 rerun with the settings of seeds 1-2 (no gradient checkpointing)
LORA_SEED0_RERUN = "lora_s0nc"
LORA_CURVE = (("256", "lora_n256"), ("1024", "lora_n1024"), ("full", "lora"))

# Edit, then LoRA: the run whose edit LoRA started from, per task
EDIT_LORA_SOURCE = {"facts": ("o1", "Opus memory edit"), "sst5inv": ("r1", "Fable swap edit")}
EDIT_LORA_SEEDS = ("edit_lora", "edit_lora_s1", "edit_lora_s2")

CONTROL_TASK = "sst5inv"
CONTROL_NAMES = ("control_random_swap", "control_swap_only")

# Ways past the SST-5-inverted plateau, three seeds each: LoRA alone with a 3x learning rate or 3x the data,
# and LoRA on the hand-written label swap
PLATEAU_RUNS = {"lr_3x": ("lora_lr3e4_s0", "lora_lr3e4_s1", "lora_lr3e4_s2"),
                "data_3x": ("lora_x3_s0", "lora_x3_s1", "lora_x3_s2"),
                "swap_only_then_lora": ("edit_swaponly_lora_s0", "edit_swaponly_lora_s1", "edit_swaponly_lora_s2")}

# SmolLM2 tasks with LoRA seeds 1-2 and edit-then-LoRA runs, and the run whose edit LoRA started from
SMOLLM_EDIT_LORA_SOURCE = {"sst5inv": "f1"}


# ---------------------------------------------------------------- result files

def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metric(entry: dict) -> dict:
    return {k: entry[k] for k in METRIC_KEYS}


def load_eval(model: str, task: str, name: str, split: str) -> dict | None:
    path = RESULTS / model / task / f"eval_{name}_{split}.json"
    return metric(read_json(path)[name]) if path.exists() else None


def evals(task: str, name: str) -> dict:
    return {split: load_eval(PRIMARY_MODEL, task, name, split) for split in SPLITS}


def load_perm(task: str, name: str) -> dict | None:
    path = RESULTS / PRIMARY_MODEL / task / f"perm_{name}_test.json"
    return read_json(path) if path.exists() else None


def general(model: str, name: str) -> dict | None:
    path = RESULTS / model / "general" / f"general_{name}.json"
    if not path.exists():
        return None
    data = read_json(path)
    wiki, mmlu = data["wikitext2"], data["mmlu"]
    return {"wiki_nll": wiki["mean_nll"],
            "wiki_ppl": wiki["perplexity"],
            "top1": wiki["top1_accuracy"],
            "scaled_nll": wiki["best_scale_nll"],
            "logit_scale": wiki["best_logit_scale"],
            "mmlu": mmlu["accuracy"],
            "mmlu_ci95": mmlu["accuracy_ci95"]}


def baselines(task: str) -> dict:
    """Base model, chance and label-prior scores per split (all three are in eval_base_<split>.json)."""
    out = {}
    for split in SPLITS:
        data = read_json(RESULTS / PRIMARY_MODEL / task / f"eval_base_{split}.json")
        out[split] = {k: metric(v) for k, v in data.items()}
    return out


# ---------------------------------------------------------------- shared with stats, tables and figures

def result_name(task: str, run: str) -> str:
    return RESULT_NAMES.get((task, run), f"wsp_{run}")


def probe_headroom(probe: dict) -> dict:
    base = probe["base_logit_dev_acc"]
    best = max(probe["probe_dev_acc"].values())
    return {"base_dev": base, "best_probe_dev": best, "headroom": best - base}


def find_run(wsp: list[dict], run: str) -> dict:
    return next(w for w in wsp if w["run"] == run)


def cost_key(model: str, task: str, run: str, role: str) -> str:
    return f"{model}/{task}/{run}/{role}"


# ---------------------------------------------------------------- Qwen tasks

def run_order(run: dict) -> tuple:
    return (run["condition"] == "preserve", PROGRAMMER_ORDER.index(run["programmer"]), run["run"])


def collateral(task: str, label: str | None) -> dict | None:
    target = COLLATERAL_TARGET.get(task)
    if target is None or label is None:
        return None
    found = load_eval(PRIMARY_MODEL, target, f"collat_{label}", "dev")
    return {"target_task": target, **found} if found else None


def preserve_collateral(run: dict) -> dict | None:
    if run.get("preserve_acc") is None:
        return None
    return {"target_task": PRESERVE_TARGET, "accuracy": run["preserve_acc"], "source": PRESERVE_SOURCE}


def wsp_result(run: dict) -> dict:
    """Run metadata plus its scores.

    Where dev was not re-measured, the dev score reported in edits.json stands in for it in evals
    (and is not repeated beside it).
    """
    task = run["task"]
    name = result_name(task, run["run"])
    meta = {k: v for k, v in run.items() if k != "reported_dev"}

    scores = None
    if run["edit"]:
        scores = evals(task, name)
        scores["dev"] = scores["dev"] or run.get("reported_dev")

    return {**meta,
            "evals": scores,
            "perm": load_perm(task, name),
            "collateral": collateral(task, COLLATERAL_LABELS.get((task, run["run"]))) or preserve_collateral(run),
            "general": general(PRIMARY_MODEL, f"wsp_{task}_{run['run']}")}


def task_summary(task: str, wsp: list[dict]) -> dict:
    lora = LORA_NAME[task]
    entry = {
        "baselines": baselines(task),
        "probe": read_json(RESULTS / PRIMARY_MODEL / task / "probe.json"),
        "perm_base": load_perm(task, "base"),
        "lora": {"evals": evals(task, lora),
                 "perm": load_perm(task, lora),
                 "collateral": collateral(task, f"lora_{task}"),
                 "general": general(PRIMARY_MODEL, f"lora_{task}")},
        "wsp": wsp,
    }
    if lora != DEFAULT_LORA:
        entry["lora_default"] = evals(task, DEFAULT_LORA)
    return entry


def test_eval(task: str, name: str, model: str = PRIMARY_MODEL) -> dict | None:
    return load_eval(model, task, name, "test")


def lora_variants() -> dict:
    return {task: {"seeds": [test_eval(task, name) for name in LORA_SEEDS],
                   "seed0_no_checkpointing": test_eval(task, LORA_SEED0_RERUN),
                   "curve": {size: test_eval(task, name) for size, name in LORA_CURVE}}
            for task in LORA_SEED_TASKS}


def edit_lora_seeds(model: str, task: str) -> dict:
    """Test accuracy and general ability of LoRA trained on top of an edit, per seed.

    The general-ability files carry the task in their name (general_edit_lora_facts_s1.json).
    """
    return {"seeds": [test_eval(task, name, model) for name in EDIT_LORA_SEEDS],
            "general": [general(model, name.replace("edit_lora", f"edit_lora_{task}")) for name in EDIT_LORA_SEEDS]}


def edit_lora() -> dict:
    return {task: {"source_edit": f"{task} {run} ({description})", **edit_lora_seeds(PRIMARY_MODEL, task)}
            for task, (run, description) in EDIT_LORA_SOURCE.items()}


def controls() -> dict:
    return {name: {f"{CONTROL_TASK}_test": test_eval(CONTROL_TASK, name)} for name in CONTROL_NAMES}


def plateau() -> dict:
    return {key: [test_eval(CONTROL_TASK, name) for name in names] for key, names in PLATEAU_RUNS.items()}


# ---------------------------------------------------------------- SmolLM2

def smollm_result(run: dict) -> dict:
    task, name = run["task"], f"wsp_{run['run']}"
    return {**run,
            "test": test_eval(task, name, SECOND_MODEL),
            "swap_only_test": test_eval(task, f"{name}_swap", SECOND_MODEL),
            "general": general(SECOND_MODEL, f"wsp_{task}_{run['run']}")}


def smollm(runs: list[dict]) -> dict:
    tasks = {}
    for task in SMOLLM_TASKS:
        tasks[task] = {
            "base": {"test": test_eval(task, "base", SECOND_MODEL)},
            "probe": probe_headroom(read_json(RESULTS / SECOND_MODEL / task / "probe.json")),
            "lora": test_eval(task, "lora", SECOND_MODEL),
            "control_random_swap": test_eval(task, "control_random_swap", SECOND_MODEL),
            "wsp": [smollm_result(r) for r in runs if r["model"] == SECOND_MODEL and r["task"] == task],
        }
        if task in SMOLLM_EDIT_LORA_SOURCE:
            tasks[task]["lora_seeds"] = [test_eval(task, name, SECOND_MODEL) for name in LORA_SEEDS]
            tasks[task]["edit_lora"] = {"source_run": SMOLLM_EDIT_LORA_SOURCE[task],
                                        **edit_lora_seeds(SECOND_MODEL, task)}
    return {"general_base": general(SECOND_MODEL, "base"), "tasks": tasks}


# ---------------------------------------------------------------- summary

def costs() -> dict:
    """Programmer and auditor API cost (USD) per run, keyed model/task/run/role."""
    return {cost_key(r["model"], r["task"], r["run"], r["role"]): r["usd"] for r in read_json(COSTS_PATH)["runs"]}


def collect() -> dict:
    meta = read_json(EDITS_PATH)
    summary = {"tasks": {},
               "incomplete": meta["incomplete"],
               "lora_train_seconds": meta["lora_train_seconds"],
               "total_params": meta["total_params"],
               "noncompliant": []}

    for task in TASKS:
        runs = sorted((r for r in meta["runs"] if r["model"] == PRIMARY_MODEL and r["task"] == task), key=run_order)
        results = [wsp_result(r) for r in runs]
        summary["tasks"][task] = task_summary(task, [w for w in results if w["compliant"]])
        summary["noncompliant"] += [w for w in results if not w["compliant"]]

    summary["general_base"] = general(PRIMARY_MODEL, "base")
    summary["lora_variants"] = lora_variants()
    summary["edit_lora"] = edit_lora()
    summary["controls"] = controls()
    summary["plateau"] = plateau()
    summary["smollm"] = smollm(meta["runs"])
    summary["costs"] = costs()
    return summary


def main() -> None:
    summary = collect()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
    for task, t in summary["tasks"].items():
        test = t["lora"]["evals"]["test"]
        print(task, "lora", round(test["accuracy"], 3),
              [(w["run"],
                w["evals"] and round(w["evals"]["test"]["accuracy"], 3),
                w["collateral"] and round(w["collateral"]["accuracy"], 3))
               for w in t["wsp"]])


if __name__ == "__main__":
    main()
