import importlib.util
from pathlib import Path

_lib = Path(__file__).resolve().parent / "_memlib.py"
_spec = importlib.util.spec_from_file_location("_memlib", _lib)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def apply(model, tok):
    _mod.apply_memory(model, Path(__file__).stem)
