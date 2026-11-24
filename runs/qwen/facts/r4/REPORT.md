# Programmer report: qwen facts r4

Final message of the programmer agent, recovered from its transcript.

**task**: facts
**run**: r4
**edit_name**: circuit (`edits/facts/r4/circuit.py`, with `circuit_config.npy`, `circuit_names.npy`, `circuit_values.npy`, `circuit_natural_tokens.npy` beside it)
**dev_acc**: 1.000 (900/900, full dev via `Lab(edit="circuit")`, two `lab.forward` passes)
**dev_nll**: 0.00055
**base_dev_acc**: 0.194 (reference given; my own base pass on dev[0:512] gave 0.170, NLL 3.82)
**passes_used**: 17 of 50
**weights_changed**: 6,182,912
**weights_fraction**: 0.0040 (0.40% of 1,543,714,304)
**kaggle_jobs**: 2 completed (kernel versions 1 and 3). Version 2 sat QUEUED and was cancelled uncharged; the coordinator re-pushed the identical build as version 3 and I only fetched it.

**mechanism**: A hand-built lookup circuit in layers 0–1, with every weight derived in closed form from the existing weights plus the name and fact table. The base model cannot know the facts, and only the final position is observable, so nothing relies on the model's own processing.
- **Embeddings (301 rows replaced)**: the 265 name sub-word pieces, the first token of each of the 30 option values, the letters A–E, and `<|im_start|>` (used as a "null" attention target). Each is solved by pseudo-inverse so its layer-0 key and value images are exactly the designed tag and code.
- **Two "gather" heads (layer 0, heads 0 and 6)**: a constant query makes every position average the codes of the name pieces seen so far.
- **One "bind" head (layer 0, head 1)**: a constant rotary positional query makes a token just after an option letter receive that letter's code; elsewhere it attends to the null token and writes nothing.
- **900 MLP neurons (layer 0)**: one per fact. A neuron fires only when the gathered name matches the person, the current token is that person's value, and a letter code is present. It writes a "flag" direction.
- **One readout head (layer 1, head 0)**: every position attends to the flagged token and copies its letter code into that letter's unembedding dual direction, at a norm (~40,700) that swamps all later layers.
- **What the passes showed**: the first real-weights run was already 100%. The stored edit also scores 2700/2700 on all of train (NLL 0.00055), so every fact is retrieved under all four available phrasings. Final-position captures matched the design: gathered name code cosine 0.9999, readout wrote 1.05× the intended vector on the true letter in every case.
- **Why it should hold for other phrasings**: the circuit keys only on name tokens and the option-list format, not on question wording. This is argued from the design, not measured beyond the four phrasings.

**failed**: Nothing failed on Kaggle. Two bugs were caught before any pass was spent, on a random-weight stand-in:
- The null token's value image leaked into the name-code coordinates, which inverted some detector thresholds; fixed by cancelling the leak exactly.
- The answer-vector norm was initially set relative to natural norms instead of my own code scale.

One weaker configuration was replaced: the first stored config (scale 407) also scored 900/900 on dev, but the tightest name pairs had less detector headroom at the final position (own-name response min 0.925). The final config uses scale 4071 (min 0.992).

**borderline** (rule 1):
- **Fact table in the weights**: a per-entity table derived from train labels (300 people × 3 attributes) is written explicitly into the 900 neurons, with the 300 names embedded as constants. Dev labels were only used to assert agreement with that table.
- **Closed-form linear algebra on weights**: pseudo-inverses and SVDs, plus two ridge "matched filters" (the bind head's positional query against ordinary-token keys, and per-person name detectors against the other 299 names). These are computed from weights and name strings only, never from task labels or forward-pass outputs, but they are regularised least-squares solves.
- **Score tables for my own heads**: I computed query·key tables over (token type × offset) as weight arithmetic to check margins. No sequence was run through the network outside `lab.forward`.
- **Local unit test on a stand-in**: I tested the construction code on a randomly initialised Qwen2-shaped network on the laptop (float32 and fp16 on CPU). No Qwen weights were ever loaded locally; I used only the tokenizer from a cached Qwen2.5-Coder checkpoint.
- **Choices made on dev**: the code scale was picked among three tested values (30×, 100×, 300× the natural layer-0 norm), all at 100% accuracy, by largest detector headroom. The logit margin (9.0 ≈ ln 4n) was chosen as the NLL-optimal gap for zero observed errors.
- **Test phrasing**: `src/wsp/tasks/facts.py`, which I was told to read, contains the test templates. I did not open the test split and did not use those templates in the design.

Files are in ``:
- `edits/facts/r4/circuit.py`
- `wsp_work/facts_r4/LOG.md`
- `wsp_work/facts_r4/step1_report.json`
- `wsp_work/facts_r4/step2_report.json`
- `results/facts/r4/wsp_ledger.json`
