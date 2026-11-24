"""Twin-2K-500: predict a real survey respondent's answer from their demographics and eight other answers.

Each example takes one respondent and one question with at least five options. The prompt shows the
respondent's demographics, eight of their other (non-demographic) answers, and five of the question's
options, one of them the respondent's real answer. People, not examples, are split into train, dev and test.
"""
import json
import random
import re
from dataclasses import dataclass

import pandas as pd

from wsp import config
from wsp.data import download
from wsp.prompts import render_user_message

SOURCE_URL = ("https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/"
              "question_catalog_and_human_response_csv/{name}")
CATALOG_PATH = config.RAW_DIR / "question_catalog.json"
RESPONSES_PATH = config.RAW_DIR / "wave1_3_response.csv"

DEMOGRAPHICS_BLOCK = "Demographics"
# Qualtrics question selectors kept: single-answer multiple choice (vertical, horizontal) and matrix rows.
SINGLE_ANSWER_MC_SELECTORS = ("SAVR", "SAHR")
MATRIX_SELECTORS = ("Likert", "Bipolar")
MAX_QUESTION_CHARS = 600
MAX_OPTION_CHARS = 160

SPLIT_SEED = 0
SAMPLE_SEED = 1
TRAIN_PEOPLE = 1400
DEV_PEOPLE = 200
TRAIN_ITEMS_PER_PERSON = 12
EVAL_ITEMS_PER_PERSON = 8
CONTEXT_FACTS_PER_PROMPT = 8

SYSTEM = "You predict how a specific real survey respondent answered. Reply with a single letter."
PERSONA_HEADER = "About the respondent:"
FACTS_HEADER = "Some of their other survey answers:"
QUESTION_HEADER = "Question:"
INSTRUCTION = "Which option did this respondent choose? Answer with the letter only."

HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------- question catalog

@dataclass(frozen=True)
class Item:
    column: str
    qid: str
    block: str
    text: str
    options: tuple[str, ...]

    @property
    def is_demographic(self) -> bool:
        return self.block.strip() == DEMOGRAPHICS_BLOCK


def clean(text: str, limit: int) -> str:
    text = WHITESPACE.sub(" ", HTML_TAG.sub(" ", text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def multiple_choice_items(q: dict) -> list[Item]:
    options = tuple(clean(o, MAX_OPTION_CHARS) for o in q.get("Options") or [])
    text = clean(q.get("QuestionText", ""), MAX_QUESTION_CHARS)
    return [Item(q["csv_columns"][0], q["QuestionID"], q["BlockName"], text, options)]


def matrix_items(q: dict) -> list[Item]:
    """One item per matrix row, sharing the matrix's column labels as options."""
    stem = clean(q.get("QuestionText", ""), MAX_QUESTION_CHARS)
    options = tuple(clean(o, MAX_OPTION_CHARS) for o in q.get("Columns") or [])
    items = []
    for row, row_id in zip(q.get("Rows") or [], q.get("RowsID") or []):
        row_text = clean(row, MAX_OPTION_CHARS)
        text = f"{stem} — {row_text}" if stem else row_text
        items.append(Item(f"{q['QuestionID']}_{row_id}", q["QuestionID"], q["BlockName"], text, options))
    return items


def load_items() -> dict[str, Item]:
    """Single-answer items keyed by their column in the responses CSV."""
    items = {}
    for q in json.loads(CATALOG_PATH.read_text()):
        selector = (q.get("Settings") or {}).get("Selector")
        if q["QuestionType"] == "MC" and selector in SINGLE_ANSWER_MC_SELECTORS:
            produced = multiple_choice_items(q)
        elif q["QuestionType"] == "Matrix" and selector in MATRIX_SELECTORS:
            produced = matrix_items(q)
        else:
            continue
        items.update({item.column: item for item in produced if item.text and len(item.options) >= 2})
    return items


# ---------------------------------------------------------------- respondents and examples

def split_people(pids: list[int]) -> dict[int, str]:
    order = sorted(pids)
    random.Random(SPLIT_SEED).shuffle(order)
    dev_end = TRAIN_PEOPLE + DEV_PEOPLE
    return {pid: "train" if i < TRAIN_PEOPLE else "dev" if i < dev_end else "test" for i, pid in enumerate(order)}


def person_answers(row: pd.Series, items: dict[str, Item]) -> dict[str, int]:
    """Column -> zero-based option index, keeping only whole-number answers inside the item's option range."""
    answers = {}
    for column, value in row.items():
        if pd.isna(value) or float(value) != int(value):
            continue
        idx = int(value) - 1
        if 0 <= idx < len(items[column].options):
            answers[column] = idx
    return answers


def sample_options(n_options: int, answer: int, rng: random.Random) -> list[int]:
    """The answer and four random distractors, in random order.

    The draws differ in order from tasks.common.shown_options; both are kept as the study's data was built.
    """
    distractors = [i for i in range(n_options) if i != answer]
    chosen = [answer] + rng.sample(distractors, config.NUM_OPTIONS - 1)
    rng.shuffle(chosen)
    return chosen


def render_body(demographics: list[tuple[str, str]], facts: list[tuple[str, str]], question: str) -> str:
    lines = [PERSONA_HEADER]
    lines += [f"- {q} {a}" for q, a in demographics]
    lines += ["", FACTS_HEADER]
    lines += [f"- {q} → {a}" for q, a in facts]
    lines += ["", QUESTION_HEADER, question]
    return "\n".join(lines)


def build() -> list[dict]:
    for path in (CATALOG_PATH, RESPONSES_PATH):
        download(SOURCE_URL.format(name=path.name), path)
    items = load_items()
    responses = pd.read_csv(RESPONSES_PATH, index_col="pid")
    responses = responses[[c for c in items if c in responses.columns]]
    assignment = split_people(list(responses.index))
    rng = random.Random(SAMPLE_SEED)
    examples = []
    for pid, row in responses.iterrows():
        split = assignment[pid]
        given = person_answers(row, items)
        demographics = [(items[c].text, items[c].options[a]) for c, a in given.items() if items[c].is_demographic]
        fact_pool = [c for c in given if not items[c].is_demographic]
        targets = [c for c in fact_pool if len(items[c].options) >= config.NUM_OPTIONS]
        per_person = TRAIN_ITEMS_PER_PERSON if split == "train" else EVAL_ITEMS_PER_PERSON
        for column in rng.sample(targets, min(per_person, len(targets))):
            item = items[column]
            facts = rng.sample([c for c in fact_pool if c != column], CONTEXT_FACTS_PER_PROMPT)
            fact_pairs = [(items[c].text, items[c].options[given[c]]) for c in facts]
            shown = sample_options(len(item.options), given[column], rng)
            body = render_body(demographics, fact_pairs, item.text)
            examples.append({
                "pid": int(pid),
                "split": split,
                "column": column,
                "block": item.block.strip(),
                "system": SYSTEM,
                "user_message": render_user_message(body, [item.options[i] for i in shown], INSTRUCTION),
                "shown_option_ids": shown,
                "label": shown.index(given[column]),
                "n_item_options": len(item.options),
            })
    return examples
