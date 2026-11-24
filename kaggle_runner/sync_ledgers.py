"""Merge every forward-pass ledger in a job's downloaded runs/ folder into the local runs/ folder.

Entries are merged as a multiset (an entry counts as many times as the larger copy holds it), so a second
Kaggle run of the same job adds its charges instead of being hidden behind a "longest copy wins" rule.
Charges from more than one job id inside one sync are reported, since they mean a job ran more than once
or two jobs of one run overlapped.

Usage: python kaggle_runner/sync_ledgers.py <downloaded runs dir> <local runs dir>
"""
import json
import sys
from collections import Counter
from pathlib import Path

LEDGER_NAME = "ledger.json"


def key(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True)


def merge(local: list[dict], remote: list[dict]) -> list[dict]:
    have = Counter(key(e) for e in local)
    merged = list(local)
    for e in remote:
        k = key(e)
        if have[k] > 0:
            have[k] -= 1
        else:
            merged.append(e)
    return merged


def main() -> None:
    src_root, dst_root = Path(sys.argv[1]), Path(sys.argv[2])
    for src in src_root.rglob(LEDGER_NAME):
        dst = dst_root / src.relative_to(src_root)
        local = json.loads(dst.read_text()) if dst.exists() else []
        remote = json.loads(src.read_text())
        merged = merge(local, remote)
        if len(merged) == len(local):
            continue
        new_jobs = {e.get("job") for e in merged[len(local):]}
        if len(new_jobs) > 1:
            print(f"WARNING {dst}: new charges from {len(new_jobs)} job ids {sorted(map(str, new_jobs))}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(merged, indent=1))
        print(f"ledger {dst.relative_to(dst_root)}: {len(local)} -> {len(merged)}")


if __name__ == "__main__":
    main()
