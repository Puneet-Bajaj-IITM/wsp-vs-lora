"""Job 1: base model on full dev and 2048 train examples, with hidden states and letter DLA."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, CAPTURE_LAYERS, TRAIN_SEED, WORK, LastPosHooks, chunks, concat_results, save_result  # noqa: E402

from wsp.harness import Lab  # noqa: E402

N_TRAIN = 2048
WORDS = [" very", " negative", " positive", " neutral", "very", "negative", "positive", "neutral"]


def main():
    lab = Lab(edit=None)
    tok = lab.tok
    from wsp.scoring import render_prompts
    ex0 = lab.examples("train")[0]
    prompt = render_prompts(tok, [ex0])[0]
    ids = tok.encode(prompt)
    print("PROMPT_TOKENS", json.dumps([tok.decode([i]) for i in ids]))
    print("WORD_IDS", {w: tok.encode(w, add_special_tokens=False) for w in WORDS})
    print("LETTER_IDS", lab.letters.tolist())

    dev_n = len(lab.examples("dev"))
    results = []
    for k, idx in enumerate(chunks(list(range(dev_n)))):
        with LastPosHooks(lab) as hooks:
            r = lab.forward("dev", idx, note=f"base dev full chunk {k}", capture_layers=CAPTURE_LAYERS)
        hd, md, rms = hooks.letter_dla()
        results.append((r, hd, md, rms))
        print("DEV", k, r["summary"])
    merged = concat_results([x[0] for x in results])
    save_result("base_dev", merged, {"head_dla": np.concatenate([x[1] for x in results]).astype(np.float16),
                                     "mlp_dla": np.concatenate([x[2] for x in results]),
                                     "rms": np.concatenate([x[3] for x in results])})
    np.save(WORK / "base_dev_logp.npy", merged["logp"])

    rng = np.random.default_rng(TRAIN_SEED)
    train_idx = sorted(rng.choice(len(lab.examples("train")), N_TRAIN, replace=False).tolist())
    np.save(WORK / "train_idx_job1.npy", np.array(train_idx))
    results = []
    for k, idx in enumerate(chunks(train_idx)):
        with LastPosHooks(lab) as hooks:
            r = lab.forward("train", idx, note=f"base train probe chunk {k}", capture_layers=CAPTURE_LAYERS)
        hd, md, rms = hooks.letter_dla()
        results.append((r, hd, md, rms))
        print("TRAIN", k, r["summary"])
    merged = concat_results([x[0] for x in results])
    save_result("base_train", merged, {"head_dla": np.concatenate([x[1] for x in results]).astype(np.float16),
                                       "mlp_dla": np.concatenate([x[2] for x in results]),
                                       "rms": np.concatenate([x[3] for x in results])})
    np.save(WORK / "base_train_logp.npy", merged["logp"])
    print("PASSES_LEFT", r["passes_left"])
    print("CACHE", sorted(p.name for p in CACHE.iterdir()))


if __name__ == "__main__":
    main()
