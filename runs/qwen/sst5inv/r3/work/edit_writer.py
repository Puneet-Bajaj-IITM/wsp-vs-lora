"""Writes small deterministic edit files that compose _steerlib primitives."""
from wsp import config

HEADER = '''"""Generated edit: {doc}"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_steerlib_r3", Path(__file__).resolve().parent / "_steerlib.py")
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

NEUTRAL_SCALE = {neutral_scale!r}
INJECT_LAYERS = {layers!r}
DIRECTION_FILE = {direction_file!r}
ALPHA = {alpha!r}
TEMPERATURE = {tau!r}


def apply(model, tok):
    lib.swap_polarity(model, tok)
    if NEUTRAL_SCALE is not None:
        lib.scale_neutral(model, tok, NEUTRAL_SCALE)
    if INJECT_LAYERS:
        print("inject", lib.inject_constant(model, INJECT_LAYERS, DIRECTION_FILE, ALPHA))
    if TEMPERATURE is not None:
        lib.set_temperature(model, TEMPERATURE)
'''


def write_edit(name: str, doc: str, neutral_scale=None, layers=(), direction_file=None, alpha=None, tau=None) -> str:
    text = HEADER.format(doc=doc, neutral_scale=neutral_scale, layers=tuple(layers),
                         direction_file=direction_file, alpha=alpha, tau=tau)
    config.EDITS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EDITS_DIR / f"{name}.py").write_text(text)
    return name
