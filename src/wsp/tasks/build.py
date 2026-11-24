"""Download the sources and build every task's splits under data/processed/<task>/.

Usage: PYTHONPATH=src python -m wsp.tasks.build
"""
from wsp.data import write_splits
from wsp.tasks import facts, sst5, twin

BUILDERS = {
    "twin": twin.build,
    "sst5": lambda: sst5.build(inverted=False),
    "sst5inv": lambda: sst5.build(inverted=True),
    "facts": facts.build,
    "facts60": lambda: facts.build(facts.SMALL_NUM_PEOPLE),
}


def main() -> None:
    for task, build in BUILDERS.items():
        write_splits(task, build())


if __name__ == "__main__":
    main()
