"""The example record shared by the SST-5 and facts tasks: five shown options, one of them correct."""
import random

from wsp import config
from wsp.prompts import render_user_message


def shown_options(n_item_options: int, true_id: int, rng: random.Random) -> list[int]:
    """Indices into the item's option list: the true one and four random distractors, in random order."""
    distractors = rng.sample([i for i in range(n_item_options) if i != true_id], config.NUM_OPTIONS - 1)
    shown = distractors + [true_id]
    rng.shuffle(shown)
    return shown


def make_example(*, system: str, body: str, instruction: str, item_options: list[str], true_id: int,
                 rng: random.Random, split: str, column: str, block: str, **extra) -> dict:
    shown = shown_options(len(item_options), true_id, rng)
    return {
        "split": split,
        "column": column,
        "block": block,
        "system": system,
        "user_message": render_user_message(body, [item_options[i] for i in shown], instruction),
        "shown_option_ids": shown,
        "label": shown.index(true_id),
        "n_item_options": len(item_options),
        **extra,
    }
