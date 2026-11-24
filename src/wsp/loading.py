"""Applying a weight edit to the base model, and loading a model with an edit, a LoRA adapter, or both.

An edit is a Python file defining `apply(model, tok)`, which changes existing weights in place. A bare name
resolves to the current run's edit folder (runs/<model>/<task>/<run>/edit/<name>.py); a name containing a
slash is a path relative to the repository root, without the .py suffix.
"""
import importlib.util
import json
from pathlib import Path

import torch

from wsp import config
from wsp.scoring import DTYPE, load_model

# An adapter trained with `train_lora.py --edit X` records X here, since it only makes sense on top of X.
ADAPTER_EDIT_FILE = "wsp_edit.json"


def edit_path(edit: str) -> Path:
    return config.ROOT / f"{edit}.py" if "/" in edit else config.EDITS_DIR / f"{edit}.py"


def apply_edit(model, tok, edit: str | None) -> None:
    if not edit:
        return
    path = edit_path(edit)
    spec = importlib.util.spec_from_file_location(f"edit_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.no_grad():
        module.apply(model, tok)


def record_adapter_edit(adapter_dir: str | Path, edit: str | None) -> None:
    (Path(adapter_dir) / ADAPTER_EDIT_FILE).write_text(json.dumps({"edit": edit}))


def recorded_edit(adapter_dir: str | Path) -> str | None:
    path = Path(adapter_dir) / ADAPTER_EDIT_FILE
    return json.loads(path.read_text())["edit"] if path.exists() else None


def load_edited(edit: str | None = None, adapter: str | None = None, dtype: torch.dtype = DTYPE):
    """Returns (tokenizer, model): the base model, then the edit, then the adapter, in that order."""
    tok, model = load_model(dtype=dtype)
    apply_edit(model, tok, edit)
    if adapter:
        trained_on = recorded_edit(adapter)
        if trained_on != edit:
            print(f"warning: adapter {adapter} was trained on edit {trained_on!r}, applying {edit!r}", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).eval()
    return tok, model
