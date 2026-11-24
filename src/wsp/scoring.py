"""Loading the base model and scoring the option letters at the assistant's first token."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsp import config
from wsp.prompts import build_chat

# Every study run used fp16 inference: the Kaggle T4 GPUs have no bf16.
DTYPE = torch.float16


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def training_dtype() -> torch.dtype:
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def load_model(model_id: str = config.MODEL_ID, dtype: torch.dtype = DTYPE):
    tok = AutoTokenizer.from_pretrained(model_id)
    # Left padding keeps every prompt's last token at position -1, where the answer letter is read.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device()).eval()
    return tok, model


def letter_token_ids(tok, letters: tuple[str, ...] = config.OPTION_LETTERS) -> list[int]:
    """Token ids of the bare letters (no leading space: the answer follows the assistant header's newline)."""
    ids = [tok.encode(letter, add_special_tokens=False) for letter in letters]
    multi = {letter: i for letter, i in zip(letters, ids) if len(i) != 1}
    if multi:
        raise ValueError(f"option letters must each encode to one token; got {multi}")
    if len({i[0] for i in ids}) != len(ids):
        raise ValueError(f"option letters share a token id: {ids}")
    return [i[0] for i in ids]


def render_prompts(tok, examples: list[dict]) -> list[str]:
    return [
        tok.apply_chat_template(build_chat(ex["user_message"], ex["system"]), tokenize=False, add_generation_prompt=True)
        for ex in examples
    ]


@torch.no_grad()
def score_and_capture(tok, model, examples: list[dict], batch_size: int = config.EVAL_BATCH_SIZE,
                      letters: tuple[str, ...] = config.OPTION_LETTERS,
                      capture_layers: tuple[int, ...] = ()) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Scores each example's answer letters and keeps the residual stream at the answer position.

    Returns:
        logp    (n, len(letters)) log-softmax over the letter logits
        hidden  {layer: (n, d_model)} for each requested layer; layer i is hidden_states[i + 1] in
                transformers' numbering, and -1 is the final-norm output that feeds the unembedding
    """
    letter_ids = torch.tensor(letter_token_ids(tok, letters), device=device())
    prompts = render_prompts(tok, examples)
    logps, captured = [], {layer: [] for layer in capture_layers}
    for start in range(0, len(prompts), batch_size):
        batch = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True).to(device())
        out = model(**batch, output_hidden_states=bool(capture_layers), logits_to_keep=1)
        logits = out.logits[:, -1, :].float()[:, letter_ids]
        logps.append(torch.log_softmax(logits, dim=-1).cpu())
        for layer in capture_layers:
            states = out.hidden_states[-1] if layer == -1 else out.hidden_states[layer + 1]
            captured[layer].append(states[:, -1, :].float().cpu())
    hidden = {layer: torch.cat(states).numpy() for layer, states in captured.items()}
    return torch.cat(logps).numpy(), hidden


def score_letters(tok, model, examples: list[dict], **kwargs) -> np.ndarray:
    """Letter log-probs (n, letters) for each example; see score_and_capture for the options."""
    return score_and_capture(tok, model, examples, **kwargs)[0]
