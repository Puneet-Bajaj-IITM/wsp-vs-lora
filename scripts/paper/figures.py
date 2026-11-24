"""The paper's figures.

Reads paper/data/summary.json (collect.py). Writes paper/figures/<name>.pdf and .png for
tradeoff (task accuracy against MMLU after the change), knowledge_map (gain over base against probe headroom)
and combination (programming, then fine-tuning).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collect import EDIT_LORA_SOURCE, ROOT, SUMMARY_PATH, find_run, probe_headroom

FIG_DIR = ROOT / "paper" / "figures"
PNG_DPI = 160

INK = "#2b2b2a"
MUTED = "#8a8985"
GRID = "#e4e3df"
STYLE = {
    "base": {"color": MUTED, "marker": "o", "label": "Base model"},
    "lora": {"color": "#2a78d6", "marker": "s", "label": "LoRA (gradient training)"},
    "bypass": {"color": "#eb6834", "marker": "^", "label": "WSP, circuit that bypasses the network"},
    "through": {"color": "#1baf7a", "marker": "D", "label": "WSP, edit through the model's own pathways"},
    "preserve": {"color": "#8b5cf6", "marker": "P", "label": "WSP told to preserve the model"},
}
MARKER_SIZE = 70
LEGEND_MARKER_SIZE = 7
TITLE_FONT = 9
AXIS_FONT = 8
NOTE_FONT = 7
SMALL_FONT = 6.5
REFERENCE_LINE_WIDTH = 1
GRID_LINE_WIDTH = 0.6

TRADEOFF_SIZE = (7.2, 3.6)
MMLU_CHANCE = 0.25
# Per panel: title, x range, and whether to mark the label-prior accuracy
TRADEOFF_PANELS = {
    "facts": ("Fictional facts", (15, 102), False),
    "twin": ("Twin-2K survey answers", (25, 50), True),
}
TRADEOFF_YLIM = (15, 68)
TRADEOFF_EDGE_WIDTH = 1.5
# Reference-line labels: height above their line, and the label-prior text's offset and height
LINE_LABEL_RISE = 1.2
PRIOR_LABEL_SHIFT = 0.6
PRIOR_LABEL_Y = 35
# Room at the bottom of the figure for its three-column legend
LEGEND_ROOM = 0.12
LEGEND_ANCHOR = (0.5, -0.02)

MAP_SIZE = (5.6, 4.2)
MAP_XLIM = (-5, 45)
MAP_YLIM = (-5, 88)
MAP_EDGE_WIDTH = 1.2
# The gain = headroom diagonal, and the position and angle of its label
DIAGONAL_END = 45
DIAGONAL_LABEL = (24, 18.5)
DIAGONAL_ANGLE = 33
TASK_SHORT = {"sst5": "SST-5", "sst5inv": "SST-5 inv.", "twin": "Twin-2K", "facts": "Facts"}
# Task labels sit at each task's highest point, nudged (in points) where they would collide
LABEL_OFFSET = {"Facts (SmolLM2)": (6, -11), "SST-5 inv.": (7, -3), "Twin-2K": (6, 2)}
DEFAULT_LABEL_OFFSET = (6, 2)

COMBO_SIZE = (6.4, 3.0)
COMBO_YLIM = (0, 105)
BAR_WIDTH = 0.26
BAR_FILL = 0.9
BAR_ALPHA = 0.35
SEED_MARKER_SIZE = 18
SEED_EDGE_WIDTH = 0.6
COMBO_LEGEND_ANCHOR = (0.5, 1.14)
COMBO_COLORS = {"edit": STYLE["through"]["color"], "lora": STYLE["lora"]["color"], "both": "#7c3aed"}
COMBO_LABELS = {"edit": "Programmed edit alone", "lora": "LoRA alone (4,000 examples)",
                "both": "Edit, then LoRA (4,000 examples)"}


# ---------------------------------------------------------------- shared drawing

def style_axes(ax) -> None:
    ax.grid(color=GRID, lw=GRID_LINE_WIDTH, zorder=0)
    ax.tick_params(labelsize=NOTE_FONT, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def mark(ax, kind: str, x: float, y: float, edge_width: float) -> None:
    st = STYLE[kind]
    ax.scatter(x, y, s=MARKER_SIZE, color=st["color"], marker=st["marker"], edgecolor="white", linewidth=edge_width,
               zorder=3)


def legend_handles(kinds) -> list:
    return [plt.Line2D([], [], color=STYLE[k]["color"], marker=STYLE[k]["marker"], ls="", markersize=LEGEND_MARKER_SIZE,
                       label=STYLE[k]["label"])
            for k in kinds]


# ---------------------------------------------------------------- tradeoff

def tradeoff_points(s: dict, task: str) -> list[tuple[str, float, float]]:
    t = s["tasks"][task]
    pts = [("base", t["baselines"]["test"]["base"]["accuracy"], s["general_base"]["mmlu"]),
           ("lora", t["lora"]["evals"]["test"]["accuracy"], t["lora"]["general"]["mmlu"])]
    for w in t["wsp"]:
        if w["evals"]:
            kind = "preserve" if w["condition"] == "preserve" else w["kind"]
            pts.append((kind, w["evals"]["test"]["accuracy"], w["general"]["mmlu"]))
    return [(kind, 100 * x, 100 * y) for kind, x, y in pts]


def draw_tradeoff_panel(ax, s: dict, task: str, title: str, xlim: tuple, mark_prior: bool) -> None:
    chance = 100 * MMLU_CHANCE
    unedited = 100 * s["general_base"]["mmlu"]
    ax.axhline(unedited, color=MUTED, lw=REFERENCE_LINE_WIDTH, ls="--", zorder=1)
    ax.axhline(chance, color=MUTED, lw=REFERENCE_LINE_WIDTH, ls=":", zorder=1)
    label_at = ax.get_yaxis_transform()
    ax.text(0.02, unedited + LINE_LABEL_RISE, "unedited model", color=MUTED, fontsize=NOTE_FONT, ha="left",
            transform=label_at)
    ax.text(0.02, chance + LINE_LABEL_RISE, f"chance ({chance:.0f}%)", color=MUTED, fontsize=NOTE_FONT, ha="left",
            transform=label_at)
    if mark_prior:
        prior = 100 * s["tasks"][task]["baselines"]["test"]["population_prior"]["accuracy"]
        ax.axvline(prior, color=MUTED, lw=REFERENCE_LINE_WIDTH, ls="-.", zorder=1)
        ax.text(prior - PRIOR_LABEL_SHIFT, PRIOR_LABEL_Y, "label prior", color=MUTED, fontsize=NOTE_FONT, rotation=90,
                ha="right")

    for kind, x, y in tradeoff_points(s, task):
        mark(ax, kind, x, y, TRADEOFF_EDGE_WIDTH)

    ax.set_title(title, fontsize=TITLE_FONT, color=INK, loc="left")
    ax.set_xlabel("Task test accuracy (%)", fontsize=AXIS_FONT, color=INK)
    ax.set_xlim(xlim)
    ax.set_ylim(*TRADEOFF_YLIM)
    style_axes(ax)


def tradeoff(s: dict):
    fig, axes = plt.subplots(1, len(TRADEOFF_PANELS), figsize=TRADEOFF_SIZE, sharey=True)
    for ax, (task, panel) in zip(axes, TRADEOFF_PANELS.items()):
        draw_tradeoff_panel(ax, s, task, *panel)
    axes[0].set_ylabel("MMLU accuracy after the change (%)", fontsize=AXIS_FONT, color=INK)
    fig.legend(handles=legend_handles(STYLE), loc="lower center", ncol=3, fontsize=NOTE_FONT, frameon=False,
               bbox_to_anchor=LEGEND_ANCHOR)
    fig.tight_layout(rect=(0, LEGEND_ROOM, 1, 1))
    return fig


# ---------------------------------------------------------------- knowledge map

def gain(accuracy: float, base: float) -> float:
    return 100 * (accuracy - base)


def map_points(s: dict) -> list[tuple[str, str, float, float]]:
    """(kind, task label, headroom, gain) for every LoRA run and every standard WSP run, on both models."""
    pts = []
    for task, t in s["tasks"].items():
        head = 100 * probe_headroom(t["probe"])["headroom"]
        base = t["baselines"]["test"]["base"]["accuracy"]
        # Every task is mapped with its default-configuration LoRA; facts reports a longer run elsewhere.
        lora = t.get("lora_default", t["lora"]["evals"])["test"]
        pts.append(("lora", TASK_SHORT[task], head, gain(lora["accuracy"], base)))
        for w in t["wsp"]:
            if w["evals"] and w["condition"] != "preserve":
                pts.append((w["kind"], TASK_SHORT[task], head, gain(w["evals"]["test"]["accuracy"], base)))

    for task, t in s["smollm"]["tasks"].items():
        label = f"{TASK_SHORT[task]} (SmolLM2)"
        head = 100 * t["probe"]["headroom"]
        base = t["base"]["test"]["accuracy"]
        if t["lora"]:
            pts.append(("lora", label, head, gain(t["lora"]["accuracy"], base)))
        for w in t["wsp"]:
            pts.append((w["kind"], label, head, gain(w["test"]["accuracy"], base)))
    return pts


def knowledge_map(s: dict):
    fig, ax = plt.subplots(figsize=MAP_SIZE)
    ax.plot([0, DIAGONAL_END], [0, DIAGONAL_END], color=MUTED, lw=REFERENCE_LINE_WIDTH, ls="--", zorder=1)
    ax.text(*DIAGONAL_LABEL, "gain = headroom", color=MUTED, fontsize=NOTE_FONT, ha="center", rotation=DIAGONAL_ANGLE)

    top: dict[str, tuple[float, float]] = {}
    for kind, task, x, y in map_points(s):
        mark(ax, kind, x, y, MAP_EDGE_WIDTH)
        if task not in top or y > top[task][1]:
            top[task] = (x, y)
    for task, (x, y) in top.items():
        ax.annotate(task, (x, y), xytext=LABEL_OFFSET.get(task, DEFAULT_LABEL_OFFSET), textcoords="offset points",
                    fontsize=SMALL_FONT, color=INK)

    ax.set_xlabel("Probe headroom: unused answer information in the base model (points)", fontsize=AXIS_FONT, color=INK)
    ax.set_ylabel("Test-accuracy gain over the base model (points)", fontsize=AXIS_FONT, color=INK)
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    style_axes(ax)
    ax.legend(handles=legend_handles(("lora", "bypass", "through")), fontsize=SMALL_FONT, frameon=False,
              loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- programming, then fine-tuning

def accuracies(ms: list[dict | None]) -> list[float]:
    return [100 * m["accuracy"] for m in ms if m]


def edit_accuracy(s: dict, task: str) -> float:
    run, _ = EDIT_LORA_SOURCE[task]
    return 100 * find_run(s["tasks"][task]["wsp"], run)["evals"]["test"]["accuracy"]


def combo_series(s: dict) -> dict[str, dict[str, list[float]]]:
    return {
        "Fictional facts": {"edit": [edit_accuracy(s, "facts")],
                            "lora": [100 * s["tasks"]["facts"]["lora_default"]["test"]["accuracy"]],
                            "both": accuracies(s["edit_lora"]["facts"]["seeds"])},
        "SST-5 inverted": {"edit": [edit_accuracy(s, "sst5inv")],
                           "lora": accuracies(s["lora_variants"]["sst5inv"]["seeds"]),
                           "both": accuracies(s["edit_lora"]["sst5inv"]["seeds"])},
    }


def combination(s: dict):
    fig, ax = plt.subplots(figsize=COMBO_SIZE)
    series = combo_series(s)
    for i, groups in enumerate(series.values()):
        for j, (key, label) in enumerate(COMBO_LABELS.items()):
            vals = groups[key]
            x = i + (j - 1) * BAR_WIDTH
            ax.bar(x, sum(vals) / len(vals), BAR_WIDTH * BAR_FILL, color=COMBO_COLORS[key], alpha=BAR_ALPHA, zorder=2,
                   label=label if i == 0 else None)
            ax.scatter([x] * len(vals), vals, s=SEED_MARKER_SIZE, color=COMBO_COLORS[key], edgecolor="white",
                       linewidth=SEED_EDGE_WIDTH, zorder=3)
    ax.set_xticks(range(len(series)), list(series), fontsize=AXIS_FONT, color=INK)
    ax.set_ylabel("Test accuracy (%)", fontsize=AXIS_FONT, color=INK)
    ax.set_ylim(*COMBO_YLIM)
    style_axes(ax)
    ax.legend(fontsize=SMALL_FONT, frameon=False, loc="upper center", ncol=len(COMBO_LABELS),
              bbox_to_anchor=COMBO_LEGEND_ANCHOR)
    fig.tight_layout()
    return fig


FIGURES = {"tradeoff": tradeoff, "knowledge_map": knowledge_map, "combination": combination}


def save_all(s: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        fig = build(s)
        fig.savefig(FIG_DIR / f"{name}.pdf")
        fig.savefig(FIG_DIR / f"{name}.png", dpi=PNG_DPI)
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    save_all(json.loads(SUMMARY_PATH.read_text()))
