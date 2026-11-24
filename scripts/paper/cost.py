"""Programmer and auditor API cost per WSP run, from the agents' transcripts (CPU only).

Reads the transcript directories given on the command line (not in the repository); writes paper/data/costs.json.
Each agent transcript (JSONL) is tagged by its opening user messages: programmers state "task = X, run = Y"
(or point at their prompt file), auditors state "task=X, run=Y". Token usage is summed per assistant turn and
priced per model.

Usage: python scripts/paper/cost.py <transcript_dir> [<transcript_dir> ...]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from collect import COSTS_PATH

TRANSCRIPT_GLOBS = ("**/*.output", "**/*.jsonl")
PROMPT_HEAD_CHARS = 400_000
OPENING_USER_TURNS = 2
PER_MILLION = 1_000_000
CACHE_WRITE_MULTIPLIER = 1.25

# USD per million tokens (Anthropic first-party list prices, Nov 2026). Fable 5.1's cache-read price is not
# listed in the reference used; 0.1x input is assumed, as for the other models.
PRICES = {
    "claude-fable-5-2": {"input": 8.0, "output": 40.0, "cache_read": 0.20},
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 1.00},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")

# The transcripts quote the prompt files where they lived when the runs happened,
# wsp_work/_prompts/<task>_<run>.txt. They are historical records, so these patterns must keep matching that old
# path and must not follow the current repository layout.
PROMPT_DIR = "_prompts/"
PROGRAMMER_PATTERNS = (
    re.compile(r"task = (\w+), run = (\w+)"),
    re.compile(r"_prompts/(?:smollm_)?([a-z0-9]+)_([a-z0-9]+)\.txt"),
)
AUDITOR_PATTERN = re.compile(r"task=(\w+), run=(\w+)")
SMOLLM_PATTERN = re.compile(r"smollm|SmolLM2", re.I)
PROGRAMMER_BRIEF = "weight programmer"
# The first brief (Twin-2K r1) predates the "task = X, run = Y" line, so it names neither.
UNNAMED_BRIEF_RUN = {"task": "twin", "run": "r1"}


def agent_key(path: Path) -> str:
    """Task outputs (a<id>.output) and subagent transcripts (agent-a<id>.jsonl) are the same agent."""
    return path.stem.removeprefix("agent-")


def events(path: Path):
    """The JSON objects in a transcript; lines that do not parse (output cut off mid-write) are skipped."""
    with path.open() as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


def first_user_text(path: Path) -> str:
    """The opening user turns; workflow agents get a relayed request first and their brief in the next turn."""
    texts = []
    for event in events(path):
        if event.get("type") == "user":
            texts.append(json.dumps((event.get("message") or {}).get("content")))
            if len(texts) == OPENING_USER_TURNS:
                break
    return " ".join(texts)[:PROMPT_HEAD_CHARS]


def tag(text: str) -> dict | None:
    """Role, task, run and base model of a WSP agent, or None for any other transcript."""
    model = "smollm" if SMOLLM_PATTERN.search(text) else "qwen"
    lowered = text.lower()
    if "auditor" in lowered:
        match = AUDITOR_PATTERN.search(text)
        return {"role": "auditor", "task": match.group(1), "run": match.group(2), "model": model} if match else None

    is_brief = PROGRAMMER_BRIEF in lowered
    if not is_brief and PROMPT_DIR not in text:
        return None
    for pattern in PROGRAMMER_PATTERNS:
        match = pattern.search(text)
        if match:
            return {"role": "programmer", "task": match.group(1), "run": match.group(2), "model": model}

    # Only a programmer brief can be the unnamed one; text that merely mentions the prompt directory
    # (such as a brief for maintaining this script) is not a programmer.
    return {"role": "programmer", **UNNAMED_BRIEF_RUN, "model": model} if is_brief else None


def usage_by_model(path: Path) -> dict[str, Counter]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for event in events(path):
        message = event.get("message") or {}
        usage = message.get("usage")
        if event.get("type") != "assistant" or not usage:
            continue
        model = event.get("requestedModel") or message.get("model")
        if model in PRICES:
            for field in USAGE_FIELDS:
                totals[model][field] += usage.get(field) or 0
    return totals


def dollars(model: str, usage: Counter) -> float:
    price = PRICES[model]
    return (usage["input_tokens"] * price["input"]
            + usage["cache_creation_input_tokens"] * price["input"] * CACHE_WRITE_MULTIPLIER
            + usage["cache_read_input_tokens"] * price["cache_read"]
            + usage["output_tokens"] * price["output"]) / PER_MILLION


def transcripts(directories: list[Path]):
    """Each agent's transcript once, even when it appears both as a task output and as a subagent log."""
    seen: set[str] = set()
    for directory in directories:
        for pattern in TRANSCRIPT_GLOBS:
            for path in directory.glob(pattern):
                agent_id = agent_key(path)
                if agent_id in seen or not path.is_file():
                    continue
                seen.add(agent_id)
                yield path


def main() -> None:
    runs: dict[tuple, dict] = {}
    for path in transcripts([Path(d) for d in sys.argv[1:]]):
        info = tag(first_user_text(path))
        if info is None:
            continue
        key = (info["model"], info["task"], info["run"], info["role"])
        entry = runs.setdefault(key, {**info, "usd": 0.0, "tokens": Counter(), "transcripts": 0, "models": Counter()})
        entry["transcripts"] += 1
        for model, usage in usage_by_model(path).items():
            entry["usd"] += dollars(model, usage)
            entry["tokens"].update(usage)
            entry["models"][model] += 1

    rows = []
    for entry in sorted(runs.values(), key=lambda e: (e["model"], e["task"], e["run"], e["role"])):
        rows.append({**{k: entry[k] for k in ("model", "task", "run", "role", "transcripts")},
                     "usd": round(entry["usd"], 2), "models": dict(entry["models"]), "tokens": dict(entry["tokens"])})
    COSTS_PATH.write_text(json.dumps({"prices_per_million": PRICES, "cache_write_multiplier": CACHE_WRITE_MULTIPLIER,
                                      "runs": rows}, indent=1))
    for row in rows:
        print(f"{row['model']:7s} {row['task']:8s} {row['run']:4s} {row['role']:10s} ${row['usd']:8.2f}  "
              f"{row['models']}")


if __name__ == "__main__":
    main()
