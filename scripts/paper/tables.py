"""Render the paper's LaTeX tables.

Reads paper/data/summary.json (collect.py) and paper/data/costs.json (cost.py).
Writes paper/tables/{tasks,main,side_effects,cost,tokens,reliability,combine,plateau,smollm,deviations}.tex.
"""
import json
import math
from collections import Counter, defaultdict

from collect import (COLLATERAL_TARGET, COSTS_PATH, EDIT_LORA_SOURCE, PRIMARY_MODEL, ROOT, SUMMARY_PATH, cost_key,
                     find_run, probe_headroom)

TABLE_DIR = ROOT / "paper" / "tables"
DASH = "--"
TASK_LABEL = {"sst5": "SST-5", "sst5inv": "SST-5 inverted", "twin": "Twin-2K", "facts": "Fictional facts"}
REGIME = {"sst5": "present, read out correctly", "sst5inv": "present, read out wrongly",
          "twin": "population tendencies only", "facts": "absent"}
SMOLLM_ORDER = ("sst5inv", "facts", "sst5")
MODEL_LABEL = {"claude-fable-5-2": "Fable 5.2", "claude-fable-5-1": "Fable 5.1", "claude-opus-5-5": "Opus 5.5"}
ROLE_LABEL = {"programmer": "Programmers", "auditor": "Auditors"}
MILLION = 1e6
# A rescaled NLL above this (or NaN) means no single temperature brings the model back, so the cell shows a dash
SCALED_NLL_MAX = 100


# ---------------------------------------------------------------- cell formatting

def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def signed_points(x: float) -> str:
    return f"{100 * x:+.1f}"


def acc(m: dict | None) -> str:
    return DASH if m is None else pct(m["accuracy"])


def acc_ci(m: dict) -> str:
    lo, hi = m["accuracy_ci95"]
    return f"{pct(m['accuracy'])} \\tiny[{pct(lo)}, {pct(hi)}]"


def seed_list(ms: list[dict | None]) -> str:
    vals = [pct(m["accuracy"]) for m in ms if m]
    return " / ".join(vals) if vals else DASH


def seed_mean(ms: list[dict | None]) -> str:
    vals = [m["accuracy"] for m in ms if m]
    return pct(sum(vals) / len(vals)) if vals else DASH


def value_range(xs: list[float]) -> str:
    low, high = pct(min(xs)), pct(max(xs))
    return low if low == high else f"{low}--{high}"


def consistency(perm: dict | None) -> str:
    return DASH if perm is None else pct(perm["fully_consistent_rate"])


def mmlu_top1(g: dict | None) -> str:
    return f"{DASH} & {DASH}" if g is None else f"{pct(g['mmlu'])} & {pct(g['top1'])}"


def general_cells(g: dict | None) -> str:
    if g is None:
        return f"{DASH} & {DASH} & {DASH}"
    scaled = g["scaled_nll"]
    rescaled = DASH if math.isnan(scaled) or scaled > SCALED_NLL_MAX else f"{scaled:.2f}"
    return f"{mmlu_top1(g)} & {rescaled}"


def programmer(w: dict) -> str:
    return f"{w['programmer']}, preserve" if w["condition"] == "preserve" else w["programmer"]


def run_label(w: dict) -> str:
    return f"WSP {w['run']}"


def latex_text(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def tabular(spec: str, header: str, rows: list[str]) -> str:
    body = " \\\\\n".join(rows)
    return (f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


# ---------------------------------------------------------------- Qwen tables

def tasks_table(s: dict) -> str:
    rows = []
    for task, t in s["tasks"].items():
        test = t["baselines"]["test"]
        probe = probe_headroom(t["probe"])
        smol = s["smollm"]["tasks"].get(task)
        smollm_cells = f"{DASH} & {DASH}"
        if smol:
            smollm_cells = f"{acc(smol['base']['test'])} & {signed_points(smol['probe']['headroom'])}"
        rows.append(f"{TASK_LABEL[task]} & {REGIME[task]} & {test['base']['n']} & {acc(test['chance'])} & "
                    f"{acc(test['population_prior'])} & {acc(test['base'])} & "
                    f"{pct(probe['best_probe_dev'])} & {signed_points(probe['headroom'])} & {smollm_cells}")
    header = ("Task & Knowledge in base model & $n_\\text{test}$ & Chance & Prior & Base & Probe & Headroom & "
              "\\multicolumn{2}{c}{SmolLM2: base / headroom}")
    return tabular("llrrrrrrrr", header, rows)


def score_row(method: str, who: str, test: dict, dev: dict | None) -> str:
    return f"{method} & {who} & {acc_ci(test)} & {test['nll']:.2f} & {acc(dev)}"


def main_table(s: dict) -> str:
    rows = []
    for task, t in s["tasks"].items():
        base = t["baselines"]
        lora = t["lora"]["evals"]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{TASK_LABEL[task]}}}}}")
        rows.append(score_row("Base", DASH, base["test"]["base"], base["dev"]["base"]))
        rows.append(score_row("Label prior", DASH, base["test"]["population_prior"], base["dev"]["population_prior"]))
        rows.append(score_row("LoRA", DASH, lora["test"], lora["dev"]))
        for w in t["wsp"]:
            if w["evals"]:
                rows.append(score_row(run_label(w), programmer(w), w["evals"]["test"], w["evals"]["dev"]))
            else:
                rows.append(f"{run_label(w)} & {programmer(w)} & \\multicolumn{{3}}{{l}}{{no edit beat base}}")
    return tabular("llllr", "Method & Programmer & Test acc. [95\\% CI] & Test NLL & Dev acc.", rows)


def side_effects_table(s: dict) -> str:
    rows = []
    base_general = general_cells(s["general_base"])
    for task, t in s["tasks"].items():
        target = COLLATERAL_TARGET.get(task)
        base_other = acc(s["tasks"][target]["baselines"]["dev"]["base"]) if target else DASH
        lora = t["lora"]
        rows.append(f"{TASK_LABEL[task]} & Base & -- & {consistency(t['perm_base'])} & {base_other} & {base_general}")
        rows.append(f" & LoRA & -- & {consistency(lora['perm'])} & {acc(lora['collateral'])} & "
                    f"{general_cells(lora['general'])}")
        for w in t["wsp"]:
            if w["evals"]:
                rows.append(f" & {run_label(w)} ({programmer(w)}) & {w['kind']} & {consistency(w['perm'])} & "
                            f"{acc(w['collateral'])} & {general_cells(w['general'])}")
    header = "Task & Method & Edit type & Consistent & Other task & MMLU & Top-1 & Rescaled NLL"
    return tabular("llllrrrr", header, rows)


def cost_table(s: dict) -> str:
    rows = []
    for task, t in s["tasks"].items():
        for w in t["wsp"]:
            fraction = f"{100 * w['weights'] / s['total_params']:.4f}\\%" if w["weights"] else DASH
            usd = s["costs"][cost_key(PRIMARY_MODEL, task, w["run"], "programmer")]
            rows.append(f"{TASK_LABEL[task]} & {w['run']} ({programmer(w)}) & {w['weights']:,} & {fraction} & "
                        f"{w['passes']} & \\${usd:,.0f} & {latex_text(w['summary'])}")
    return tabular("llrrrrp{5.4cm}", "Task & Run & Weights changed & Fraction & Passes & API cost & Mechanism", rows)


def tokens_table(costs: dict) -> str:
    """Token use and cost summed per role and model (each run counted under its first model)."""
    tokens, usd, count = defaultdict(Counter), Counter(), Counter()
    for r in costs["runs"]:
        key = (r["role"], next(iter(r["models"])))
        tokens[key].update(r["tokens"])
        usd[key] += r["usd"]
        count[key] += 1

    rows = []
    for key in sorted(tokens, key=lambda k: (list(ROLE_LABEL).index(k[0]), k[1])):
        role, model = key
        t = tokens[key]
        rows.append(f"{ROLE_LABEL[role]} & {MODEL_LABEL[model]} & {count[key]} & "
                    f"{t['output_tokens'] / MILLION:.2f} & {t['cache_read_input_tokens'] / MILLION:.1f} & "
                    f"{t['cache_creation_input_tokens'] / MILLION:.1f} & \\${usd[key]:,.0f}")
    header = "Role & Model & Runs & Output (M) & Cache read (M) & Cache write (M) & Cost"
    return tabular("llrrrrr", header, rows)


def wsp_range(wsp: list[dict]) -> str:
    accs = [w["evals"]["test"]["accuracy"] for w in wsp if w["evals"]]
    return f"{pct(min(accs))}--{pct(max(accs))}"


def reliability_table(s: dict) -> str:
    rows = []
    for task, v in s["lora_variants"].items():
        curve = v["curve"]
        rows.append(f"{TASK_LABEL[task]} & {seed_list(v['seeds'])} & {acc(curve['256'])} & {acc(curve['1024'])} & "
                    f"{acc(curve['full'])} & {wsp_range(s['tasks'][task]['wsp'])}")
    header = "Task & LoRA seeds 0 / 1 / 2 & $n=256$ & $n=1{,}024$ & full pool & WSP (all runs)"
    return tabular("lccccc", header, rows)


def combine_table(s: dict) -> str:
    facts = s["tasks"]["facts"]
    # Facts has one default-configuration LoRA and one 10x run; SST-5 inverted has three seeds of each, longer at 3x.
    lora_alone = {"facts": acc(facts["lora_default"]["test"]),
                  "sst5inv": seed_list(s["lora_variants"]["sst5inv"]["seeds"])}
    long_lora = {"facts": f"{acc(facts['lora']['evals']['test'])}$^\\ddagger$",
                 "sst5inv": f"{seed_list(s['plateau']['data_3x'])}$^\\S$"}

    rows = []
    for task, (run, description) in EDIT_LORA_SOURCE.items():
        edit = find_run(s["tasks"][task]["wsp"], run)
        combined = s["edit_lora"][task]
        combined_mmlu = " / ".join(pct(g["mmlu"]) for g in combined["general"])
        rows.append(f"{TASK_LABEL[task]} & {description} ({run}) & {acc(edit['evals']['test'])} & {lora_alone[task]} & "
                    f"{seed_list(combined['seeds'])} & {combined_mmlu} & {long_lora[task]}")
    header = ("Task & Starting edit & Edit alone & LoRA alone & Edit, then LoRA & MMLU, edit then LoRA & "
              "LoRA alone, more data")
    return tabular("llccccc", header, rows)


def plateau_table(s: dict) -> str:
    """Every way we got LoRA past the SST-5-inverted plateau, on both models (start = accuracy before LoRA)."""
    t = s["tasks"]["sst5inv"]
    run, description = EDIT_LORA_SOURCE["sst5inv"]
    smol = s["smollm"]["tasks"]["sst5inv"]
    smol_run = smol["edit_lora"]["source_run"]
    base, default = t["baselines"]["test"]["base"], "4,000 examples"
    groups = {
        "Qwen2.5-1.5B": [
            ("Base model", base, default, s["lora_variants"]["sst5inv"]["seeds"]),
            ("Base model", base, "4,000 examples, learning rate $3\\times10^{-4}$", s["plateau"]["lr_3x"]),
            ("Base model", base, "12,000 examples", s["plateau"]["data_3x"]),
            (f"{description} ({run})", find_run(t["wsp"], run)["evals"]["test"], default,
             s["edit_lora"]["sst5inv"]["seeds"]),
            ("Hand-written label swap", s["controls"]["control_swap_only"]["sst5inv_test"], default,
             s["plateau"]["swap_only_then_lora"]),
        ],
        "SmolLM2-1.7B": [
            ("Base model", smol["base"]["test"], default, smol["lora_seeds"]),
            (f"Fable edit ({smol_run})", find_run(smol["wsp"], smol_run)["test"], default, smol["edit_lora"]["seeds"]),
        ],
    }
    rows = []
    for model, entries in groups.items():
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{model}}}}}")
        rows += [f"\\quad {start} & {acc(before)} & {training} & {seed_list(seeds)} & {seed_mean(seeds)}"
                 for start, before, training, seeds in entries]
    return tabular("lclcc", "LoRA starts from & Start & LoRA training & Seeds 0 / 1 / 2 & Mean", rows)


def deviations_table(s: dict) -> str:
    rows = []
    for w in s["noncompliant"]:
        rows.append(f"{TASK_LABEL[w['task']]} & {w['run']} ({w['programmer']}) & {w['kind']} & "
                    f"{acc(w['evals']['test'])} & {acc(w['evals']['dev'])} & {acc(w['collateral'])} & "
                    f"{mmlu_top1(w['general'])} & {w['weights']:,} & {w['passes']}")
    header = "Task & Run & Edit type & Test & Dev & Other task & MMLU & Top-1 & Weights & Passes"
    return tabular("llllrrrrrr", header, rows)


# ---------------------------------------------------------------- SmolLM2

def smollm_table(s: dict) -> str:
    sm = s["smollm"]
    rows = [f"Base model & -- & -- & {mmlu_top1(sm['general_base'])}"]
    for task in SMOLLM_ORDER:
        t = sm["tasks"][task]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{TASK_LABEL[task]}}}: base {acc(t['base']['test'])}, "
                    f"probe headroom {signed_points(t['probe']['headroom'])}}}")
        if t.get("lora_seeds"):
            rows.append(f"\\quad LoRA (seeds 0 / 1 / 2) & -- & {seed_list(t['lora_seeds'])} & -- & --")
        elif t["lora"]:
            rows.append(f"\\quad LoRA (seed 0) & -- & {acc(t['lora'])} & -- & --")
        if t["control_random_swap"]:
            rows.append(f"\\quad Control: swap two unrelated words & -- & {acc(t['control_random_swap'])} & -- & --")
        for w in t["wsp"]:
            if w["swap_only_test"]:
                rows.append(f"\\quad {run_label(w)}, embedding swap alone & through & "
                            f"{acc(w['swap_only_test'])} & -- & --")
            rows.append(f"\\quad {run_label(w)}, final edit & {w['kind']} & {acc(w['test'])} & "
                        f"{mmlu_top1(w['general'])}")
        if t.get("edit_lora"):
            combined = t["edit_lora"]
            rows.append(f"\\quad WSP {combined['source_run']}, then LoRA (seeds 0 / 1 / 2) & -- & "
                        f"{seed_list(combined['seeds'])} & {value_range([g['mmlu'] for g in combined['general']])} & "
                        f"{value_range([g['top1'] for g in combined['general']])}")
    return tabular("llccc", "Method & Edit type & Test acc. & MMLU & Top-1", rows)


def main() -> None:
    s = json.loads(SUMMARY_PATH.read_text())
    costs = json.loads(COSTS_PATH.read_text())
    tables = {"tasks": tasks_table(s),
              "main": main_table(s),
              "side_effects": side_effects_table(s),
              "cost": cost_table(s),
              "tokens": tokens_table(costs),
              "reliability": reliability_table(s),
              "combine": combine_table(s),
              "plateau": plateau_table(s),
              "smollm": smollm_table(s),
              "deviations": deviations_table(s)}
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, tex in tables.items():
        (TABLE_DIR / f"{name}.tex").write_text(tex)
        print("wrote", name)


if __name__ == "__main__":
    main()
