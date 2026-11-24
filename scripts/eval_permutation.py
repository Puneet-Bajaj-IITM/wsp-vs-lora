"""Position bias versus understanding: score each question under all five cyclic rotations of its options.

A model that understands the question picks the same option text wherever it appears; one that relies on
position picks the same letter. Writes results/<model>/<task>/perm_<name>_<split>.json with:
    single_order_accuracy    accuracy over all rotations
    perm_averaged_accuracy   accuracy after averaging each option text's probability over rotations
    fully_consistent_rate    share of questions where all five rotations pick the same option text
    mean_modal_share         average share of rotations agreeing with the most common pick
    pred_rate_by_slot_ABCDE  how often each letter is predicted
"""
import argparse
import json
import random

import numpy as np

from wsp import config
from wsp.data import load_split
from wsp.loading import load_edited
from wsp.prompts import OPTIONS_HEADER
from wsp.scoring import score_letters

SUBSET_SIZE = 300
SUBSET_SEED = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--name", required=True, help="label for the result file")
    p.add_argument("--edit", default=None, help="edit applied before --adapter")
    p.add_argument("--adapter", default=None, help="PEFT adapter directory")
    p.add_argument("--n", type=int, default=SUBSET_SIZE, help="number of questions")
    return p.parse_args()


def rotation(shift: int) -> list[int]:
    return [(i + shift) % config.NUM_OPTIONS for i in range(config.NUM_OPTIONS)]


def split_message(message: str) -> tuple[str, list[str], str]:
    """(text before the option list, option texts, text after it)."""
    head, rest = message.split(f"{OPTIONS_HEADER}\n", 1)
    block, tail = rest.split("\n\n", 1)
    options = [line.split(". ", 1)[1] for line in block.splitlines()]
    return head, options, tail


def rotated_examples(ex: dict) -> list[dict]:
    head, options, tail = split_message(ex["user_message"])
    rotated = []
    for shift in range(config.NUM_OPTIONS):
        order = rotation(shift)
        lines = [f"{letter}. {options[j]}" for letter, j in zip(config.OPTION_LETTERS, order)]
        message = f"{head}{OPTIONS_HEADER}\n" + "\n".join(lines) + f"\n\n{tail}"
        rotated.append({**ex, "user_message": message, "label": order.index(ex["label"])})
    return rotated


def analyse(logp: np.ndarray, questions: list[dict]) -> dict:
    k = config.NUM_OPTIONS
    probs = np.exp(logp.reshape(len(questions), k, k))
    per_option = np.zeros((len(questions), k))
    picked_option = np.zeros((len(questions), k), dtype=int)
    for shift in range(k):
        order = rotation(shift)
        per_option[:, order] += probs[:, shift, :]
        picked_option[:, shift] = np.array(order)[probs[:, shift, :].argmax(1)]
    labels = np.array([e["label"] for e in questions])
    modal_share = np.array([np.bincount(row, minlength=k).max() / k for row in picked_option])
    slot_rate = np.bincount(probs.argmax(2).ravel(), minlength=k) / probs[:, :, 0].size
    return {
        "n_questions": len(questions),
        "single_order_accuracy": float((picked_option == labels[:, None]).mean()),
        "perm_averaged_accuracy": float((per_option.argmax(1) == labels).mean()),
        "fully_consistent_rate": float((modal_share == 1.0).mean()),
        "mean_modal_share": float(modal_share.mean()),
        "pred_rate_by_slot_ABCDE": [round(float(x), 3) for x in slot_rate],
    }


def main() -> None:
    args = parse_args()
    questions = random.Random(SUBSET_SEED).sample(load_split(args.split), args.n)
    rotated = [r for ex in questions for r in rotated_examples(ex)]
    tok, model = load_edited(args.edit, args.adapter)
    result = analyse(score_letters(tok, model, rotated), questions)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"perm_{args.name}_{args.split}.json").write_text(json.dumps(result, indent=2))
    print(args.name, json.dumps(result))


if __name__ == "__main__":
    main()
