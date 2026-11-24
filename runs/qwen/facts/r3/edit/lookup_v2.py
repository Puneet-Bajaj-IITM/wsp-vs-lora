"""lookup_v2: hand-built fact lookup (value-token codes, slot head, person neurons, answer head).

The person -> value table (300 x 3, read off the train labels) and the per-person key directions (class means of
answer-position activations) are stored beside this file; see _lookup_lib.py for the mechanism.
"""
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = str(HERE / "lookup_v2")
CONSTANTS = ("value_tokens", "prompt_tokens", "mu", "scalars", "gate", "bias_dir", "table", "gain", "keys_z")
NIBBLE_OFFSET = 8


def unpack_keys(blob: np.ndarray, rows: int) -> np.ndarray:
    packed = np.frombuffer(zlib.decompress(blob.tobytes()), dtype=np.uint8).reshape(rows, -1)
    out = np.empty((rows, packed.shape[1] * 2), dtype=np.float32)
    out[:, 0::2] = (packed >> 4).astype(np.float32) - NIBBLE_OFFSET
    out[:, 1::2] = (packed & 15).astype(np.float32) - NIBBLE_OFFSET
    return out


def apply(model, tok):
    sys.path.insert(0, str(HERE))
    import _lookup_lib as lib
    consts = lib.load_consts(PREFIX, CONSTANTS)
    consts["keys"] = unpack_keys(consts["keys_z"], len(consts["table"]))
    gain, lib.CODE_AMPLITUDE, lib.SLOT_AMPLITUDE = (float(x) for x in consts["gain"])
    lib.MARKER_AMPLITUDE = lib.CODE_AMPLITUDE
    return lib.apply_stages(model, lib.prepare(model, tok, consts), consts, logit_gain=gain)
