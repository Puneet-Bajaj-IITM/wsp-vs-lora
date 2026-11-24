"""Generated edit: swap + {'layers': (17, 18, 19, 20), 'direction_file': 'neutral_dir_h20.npy', 'alpha': 1.5, 'tau': 0.25}"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_steerlib_r3", Path(__file__).resolve().parent / "_steerlib.py")
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

NEUTRAL_SCALE = None
INJECT_LAYERS = (17, 18, 19, 20)
DIRECTION_FILE = 'neutral_dir_h20.npy'
ALPHA = 1.5
TEMPERATURE = 0.25


def apply(model, tok):
    lib.swap_polarity(model, tok)
    if NEUTRAL_SCALE is not None:
        lib.scale_neutral(model, tok, NEUTRAL_SCALE)
    if INJECT_LAYERS:
        print("inject", lib.inject_constant(model, INJECT_LAYERS, DIRECTION_FILE, ALPHA))
    if TEMPERATURE is not None:
        lib.set_temperature(model, TEMPERATURE)
