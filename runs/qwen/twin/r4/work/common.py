"""Shared indexing of items and (item, option) pairs, and the stratified exploration sample."""
import random
from collections import defaultdict

EXPLORE_SEED = 0
PASS_SIZE = 512


def item_index(train: list[dict]) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Sorted item columns, column -> item id, column -> first pair id (pairs are item x option id)."""
    n_opts = {}
    for ex in train:
        n_opts[ex["column"]] = ex["n_item_options"]
    cols = sorted(n_opts)
    offset, pair0 = 0, {}
    for c in cols:
        pair0[c] = offset
        offset += n_opts[c]
    return cols, {c: i for i, c in enumerate(cols)}, pair0, n_opts, offset


def stratified_indices(train: list[dict], n_total: int, seed: int = EXPLORE_SEED, skip: set | None = None) -> list[int]:
    """Round-robin over items so that every item gets (nearly) equal coverage."""
    rng = random.Random(seed)
    by_item = defaultdict(list)
    for i, ex in enumerate(train):
        if not skip or i not in skip:
            by_item[ex["column"]].append(i)
    for v in by_item.values():
        rng.shuffle(v)
    order, depth = [], 0
    cols = sorted(by_item)
    while len(order) < n_total:
        added = False
        for c in cols:
            if depth < len(by_item[c]) and len(order) < n_total:
                order.append(by_item[c][depth])
                added = True
        depth += 1
        if not added:
            break
    rng.shuffle(order)
    return order


def option_texts(ex: dict) -> list[str]:
    block = ex["user_message"].rsplit("\n\nOptions:\n", 1)[1].split("\n\nWhich option", 1)[0]
    return [line[3:] for line in block.split("\n")]
