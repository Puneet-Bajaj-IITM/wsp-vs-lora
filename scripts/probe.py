"""Knowledge-presence probe: how much of the label a linear read-out of the base model can find.

A logistic-regression probe is fitted per layer on the residual stream at the answer position (train sample)
and scored on a dev sample, next to the model's own letter accuracy. The best layer's probe accuracy minus
the base accuracy is the "headroom" the paper uses to place each task. Writes results/<model>/<task>/probe.json.
"""
import argparse
import json
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wsp import config
from wsp.data import load_split
from wsp.scoring import device, letter_token_ids, load_model, render_prompts

PROBE_DEPTH_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
FINAL_NORM = -1
N_TRAIN = 2048
N_DEV = 1024
SAMPLE_SEED = 5
PROBE_C = 0.05
PROBE_MAX_ITER = 3000
# Halved, together with the hooks below, after the SmolLM2 probe ran out of GPU memory.
PROBE_BATCH_DIVISOR = 2


def probe_layers(model) -> tuple[int, ...]:
    """Blocks at fixed fractions of depth; for Qwen2.5-1.5B (28 blocks) these are 6, 13, 20 and 27."""
    n = model.config.num_hidden_layers
    return tuple(round(n * f) - 1 for f in PROBE_DEPTH_FRACTIONS)


def last_token_hooks(model, layers: tuple[int, ...], store: dict[int, list]) -> list:
    """Forward hooks that keep only the last position of each chosen block's output and of the final norm.

    Asking the model for all hidden states instead keeps every layer at every position, which ran out of
    memory on SmolLM2.
    """
    def keep(key):
        def hook(_module, _inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            store[key].append(states[:, -1, :].float().cpu())
        return hook

    handles = [model.model.layers[layer].register_forward_hook(keep(layer)) for layer in layers]
    handles.append(model.model.norm.register_forward_hook(keep(FINAL_NORM)))
    return handles


@torch.no_grad()
def capture(tok, model, examples: list[dict]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Returns the letter log-probs (n, 5) and {layer: (n, d_model)} at the answer position."""
    letters = torch.tensor(letter_token_ids(tok), device=device())
    prompts = render_prompts(tok, examples)
    layers = probe_layers(model)
    logps, hidden = [], {layer: [] for layer in (*layers, FINAL_NORM)}
    handles = last_token_hooks(model, layers, hidden)
    batch_size = max(1, config.EVAL_BATCH_SIZE // PROBE_BATCH_DIVISOR)
    try:
        for start in range(0, len(prompts), batch_size):
            batch = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True).to(device())
            out = model(**batch, logits_to_keep=1)
            logps.append(torch.log_softmax(out.logits[:, -1, :].float()[:, letters], -1).cpu())
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(logps).numpy(), {k: torch.cat(v).numpy() for k, v in hidden.items()}


def probe_accuracy(x_train, y_train, x_dev, y_dev) -> float:
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER).fit(scaler.transform(x_train), y_train)
    return float((clf.predict(scaler.transform(x_dev)) == y_dev).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-dev", type=int, default=N_DEV)
    args = p.parse_args()

    rng = random.Random(SAMPLE_SEED)
    train, dev = load_split("train"), load_split("dev")
    train = rng.sample(train, min(args.n_train, len(train)))
    dev = rng.sample(dev, min(args.n_dev, len(dev)))

    tok, model = load_model()
    train_logp, train_hidden = capture(tok, model, train)
    dev_logp, dev_hidden = capture(tok, model, dev)
    y_train = np.array([e["label"] for e in train])
    y_dev = np.array([e["label"] for e in dev])
    result = {
        "task": config.TASK,
        "n_train": len(train),
        "n_dev": len(dev),
        "base_logit_dev_acc": float((dev_logp.argmax(1) == y_dev).mean()),
        "probe_dev_acc": {str(k): probe_accuracy(train_hidden[k], y_train, dev_hidden[k], y_dev) for k in train_hidden},
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "probe.json").write_text(json.dumps(result, indent=2))
    print("probe", json.dumps(result))


if __name__ == "__main__":
    main()
