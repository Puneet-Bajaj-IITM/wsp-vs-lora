"""Reading and writing task splits, and downloading source files once into data/raw/."""
import json
import time
import urllib.request
from pathlib import Path

from wsp import config

SPLITS = ("train", "dev", "test")
DOWNLOAD_RETRIES = 5
RETRY_SECONDS = 3


def download(url: str, path: Path) -> Path:
    """Returns path, fetching url into it first if it is not there yet.

    Hugging Face occasionally refuses bursts of requests, so a failed fetch is retried with a growing wait.
    """
    if path.exists():
        return path
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            data = urllib.request.urlopen(url).read()
            break
        except OSError:
            if attempt == DOWNLOAD_RETRIES - 1:
                raise
            time.sleep(RETRY_SECONDS * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_split(split: str) -> list[dict]:
    return load_jsonl(config.PROCESSED_DIR / f"{split}.jsonl")


def write_splits(task: str, examples: list[dict]) -> None:
    out_dir = config.PROCESSED_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [e for e in examples if e["split"] == split]
        (out_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(task, split, len(rows))
