"""Name parts (first name + three surname syllables) per person id, replaying the task builder's generator."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wsp.tasks.facts import FIRST_NAMES, NUM_PEOPLE, SEED, SYLLABLES  # noqa: E402

NUM_PARTS = 4


def person_parts() -> np.ndarray:
    """(people, 4) ints: first-name id, then syllable ids offset so every (slot, syllable) is a distinct part."""
    rng = random.Random(SEED)
    parts: dict[str, tuple] = {}
    while len(parts) < NUM_PEOPLE:
        syl = rng.sample(SYLLABLES, 2) + [rng.choice(SYLLABLES)]
        first = rng.choice(FIRST_NAMES)
        name = f"{first} {''.join(syl).capitalize()}"
        parts.setdefault(name, (FIRST_NAMES.index(first),) + tuple(
            len(FIRST_NAMES) + i * len(SYLLABLES) + SYLLABLES.index(s) for i, s in enumerate(syl)))
    return np.array([parts[name] for name in sorted(parts)])
