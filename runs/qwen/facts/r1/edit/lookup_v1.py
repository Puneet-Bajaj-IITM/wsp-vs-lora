"""Hand-compiled fact lookup circuit (layers 0-1). Writes a label-derived per-person fact table into weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _lk_apply


def apply(model, tok):
    _lk_apply.run(model, tok)
