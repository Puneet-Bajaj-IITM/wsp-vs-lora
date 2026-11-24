"""Held-out general-ability data: the WikiText-2 (raw) test text and a fixed sample of the MMLU test set."""
import random

import pandas as pd

from wsp import config
from wsp.data import download

CACHE_DIR = config.RAW_DIR / "general"
WIKITEXT_URL = "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-2-raw-v1/test/0.parquet"
MMLU_URL = "https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test/0.parquet"
WIKITEXT_FILE = "wikitext2_raw_test.parquet"
MMLU_FILE = "mmlu_all_test.parquet"
WIKITEXT_JOINER = "\n\n"
MMLU_SEED = 0


def wikitext_test_text() -> str:
    frame = pd.read_parquet(download(WIKITEXT_URL, CACHE_DIR / WIKITEXT_FILE))
    return WIKITEXT_JOINER.join(frame["text"].tolist())


def mmlu_sample(n: int) -> list[dict]:
    frame = pd.read_parquet(download(MMLU_URL, CACHE_DIR / MMLU_FILE))
    rows = random.Random(MMLU_SEED).sample(range(len(frame)), min(n, len(frame)))
    return [
        {
            "question": frame.at[i, "question"],
            "subject": frame.at[i, "subject"],
            "choices": [str(c) for c in frame.at[i, "choices"]],
            "answer": int(frame.at[i, "answer"]),
        }
        for i in rows
    ]
