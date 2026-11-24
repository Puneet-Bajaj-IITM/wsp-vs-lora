"""Closed-form fact lookup circuit (layers 0-1). See _hj_build.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hj_build import install  # noqa: E402
from _hj_config import HijackConfig  # noqa: E402

CONFIG = HijackConfig()


def apply(model, tok):
    install(model, tok, CONFIG)
