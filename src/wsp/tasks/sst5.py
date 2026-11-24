"""SST-5 sentiment, in two versions with identical prompts.

sst5    : the label is the true sentiment (knowledge present, read out correctly).
sst5inv : the label is the mirrored sentiment (knowledge present, read out wrongly by the base model).
"""
import random

import pandas as pd

from wsp import config
from wsp.data import download
from wsp.tasks.common import make_example

SOURCE_URL = "https://huggingface.co/api/datasets/SetFit/sst5/parquet/default/{split}/0.parquet"
SOURCE_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]
SYSTEM = "You classify the sentiment of movie review excerpts. Reply with a single letter."
INSTRUCTION = "Which option describes the sentiment of this review? Answer with the letter only."
SEED = 11
CACHE_DIR = config.RAW_DIR / "sst5"


def build(inverted: bool) -> list[dict]:
    task = "sst5inv" if inverted else "sst5"
    rng = random.Random(SEED)
    examples = []
    for split, source in SOURCE_SPLITS.items():
        frame = pd.read_parquet(download(SOURCE_URL.format(split=source), CACHE_DIR / f"{source}.parquet"))
        for i, row in enumerate(frame.itertuples()):
            label = int(row.label)
            target = len(LABELS) - 1 - label if inverted else label
            examples.append(make_example(
                system=SYSTEM,
                body=f"Review: {row.text.strip()}",
                instruction=INSTRUCTION,
                item_options=LABELS,
                true_id=target,
                rng=rng,
                split=split,
                column=task,
                block=LABELS[label],
                pid=i,
            ))
    return examples
