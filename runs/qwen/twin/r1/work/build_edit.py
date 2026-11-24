"""Write an edit module plus its constant files into edits/ from offline-derived arrays."""
from pathlib import Path

import numpy as np

from wsp import config

EDITS_DIR = config.ROOT / "edits"
TEMPLATE = Path(__file__).parent / "edit_template.py"


def build(name: str, alpha: float, wu_delta: np.ndarray, doc: str) -> Path:
    assert wu_delta.shape == (config.NUM_OPTIONS, 1536), wu_delta.shape
    EDITS_DIR.mkdir(exist_ok=True)
    np.save(EDITS_DIR / f"{name}_alpha.npy", np.array(alpha, dtype=np.float32))
    np.save(EDITS_DIR / f"{name}_wu_delta.npy", wu_delta.astype(np.float32))
    path = EDITS_DIR / f"{name}.py"
    path.write_text(TEMPLATE.read_text().replace("{doc}", doc))
    return path
