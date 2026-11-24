"""Generated edit: swap + {'tau': 0.2}"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_steerlib_r3", Path(__file__).resolve().parent / "_steerlib.py")
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

NEUTRAL_SCALE = None
INJECT_LAYERS = ()
DIRECTION_FILE = None
ALPHA = None
TEMPERATURE = 0.2


def apply(model, tok):
    lib.swap_polarity(model, tok)
    if NEUTRAL_SCALE is not None:
        lib.scale_neutral(model, tok, NEUTRAL_SCALE)
    if INJECT_LAYERS:
        print("inject", lib.inject_constant(model, INJECT_LAYERS, DIRECTION_FILE, ALPHA))
    if TEMPERATURE is not None:
        lib.set_temperature(model, TEMPERATURE)
