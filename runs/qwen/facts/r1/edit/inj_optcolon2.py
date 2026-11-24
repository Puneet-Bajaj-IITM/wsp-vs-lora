import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _injlib


def apply(model, tok):
    _injlib.apply_global(model, tok, ('Options', ':\n'), 2.0)
