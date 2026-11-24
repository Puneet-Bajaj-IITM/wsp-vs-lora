# twin / r2 log

## Local data analysis (no passes)
- train 16800 / dev 1600; 92 items. Blocks: Personality 38%, Cognitive tests 32%, False consensus 12.5%, Economic 9%, small heuristics blocks 9%.
- Dev accuracy of label-frequency predictors built from train: per-item prior 0.478; a context-free
  "option string -> P(chosen | shown)" table 0.426 (cognitive 0.577, personality 0.30, FC 0.356, econ 0.537).
- Cognitive items: modal answer = the correct answer (synonym/antonym), so "answer the question correctly" is a label-free rule there.
- Hypotheses: (H1) base has a letter bias and is over-confident (NLL 2.67 > chance 1.609) -> calibration helps NLL;
  (H2) the answer-position residual encodes which letter holds which option content -> a linear read-out of
  class-mean differences can push the popular option; (H3) editing embeddings of non-letter tokens is "free"
  (scoring only uses the 5 letter rows of the tied matrix).

## Job 1 (passes 1-4): base on 2048 random train examples, residual capture at 13 layers
- Result (2048 train ex.): base acc 0.316, NLL 2.568; cognitive 0.50, personality 0.215, FC 0.26, econ 0.22.
  Letter bias small (pred hist .14/.22/.27/.26/.11); temperature alone fixes NLL (T=4 -> 1.55) but not accuracy.
- Base picks "Agree strongly" on 47% of Likert items (true 18%) and the two "Strongly ..." options on FC: a content bias.
- H2 rejected: label-centroid read-out at any layer (4..26, final) gives <= 0.29 on held-out; the letter that
  holds a given option string is only weakly decodable at the answer position (0.28-0.56 vs 0.2 chance).
  => a read-out at the answer position cannot inject the prior; the option->letter binding has to be built.

## Design: compile a prior circuit into layers 0-1 (no passes; verified offline)
- Layer 0 inputs are raw embeddings, so everything there can be made exact. Private residual coordinates
  (lowest-energy embedding columns) carry token features; RoPE gives position kernels.
- Hop 1 (layer 0, KV group 0): option token t attends to the letter opening its line with weight sigmoid(G_t);
  the competitor is the letter's own dot, whose key is rotated by one RoPE step so both see the same kernel value.
  G_t comes from a train-derived token table (4 channels = 4 heads). A 5th head with a dot bonus cancels
  "stray" capital-letter tokens (no dot after them). 6th head = selector: nearest question marker -> channel.
- Hop 2 (layer 1, KV group 0): head m averages channel m's letter bus over the prompt if selected (else a null
  token); result goes to 5 private coordinates -> final-norm gain -> letter rows of the tied embedding.
- Kernels fitted to clipped ramps over relative offset only (label-free ridge least squares). First version
  wrongly assumed RoPE base 1e4; Qwen2.5 uses 1e6 (found in config) -> redesigned.
- Tables (TRAIN label counts): per item, option choice rates spread over option tokens ("sequential residual
  means"); items grouped into 4 channels (greedy + reassignment by agreement with the item's own rates);
  one marker token per non-default item. Offline emulation of the circuit on dev prompts: prior-argmax
  accuracy 0.455 (held-out train people 0.45); item-level prior is 0.478.
- Unit test on a RANDOM-weight Qwen2-shaped toy (3 layers, CPU): bus values match the emulation exactly in
  float32 and within 0.02 in float16. (The real model is never loaded locally.)

## Job 2: first run of edits/r2/prior_circuit.py on the real model
- Passes 5-10. Weights changed 14,201,119 / 1,543,714,304 (0.92%).
- Pass A (train 512, gain 1, T 1): acc 0.201, predicts "B" 99% -> the surgery (6 of 12 heads replaced in layers 0
  and 1, 46 residual coordinates taken) destroys the model's own answer. Model logits are useless after the edit.
- Read-out grid on pass A (model temperature T x prior gain; prior term reconstructed from the captured out-bus):
  best = ignore the model (T=1000) with gain 3.2: 0.449. Pass B (fresh train 512, same setting): 0.451, NLL 1.382.
- Full dev with in-script T=1000, gain 3.2: acc 0.450, NLL 1.390 (base 0.294 / 2.674). Emulation predicted 0.455.
  => the compiled circuit computes the prior on the real model as designed; the model itself contributes nothing.

## Offline analysis after job 2 (no passes)
- Emulated prior vs measured out-bus on pass A: correlation 0.946, argmax agreement 93% (differences come from the
  per-token 1/rms at layer 1 and the 1/N average). Measured accuracy 0.451 vs emulated 0.441 on those 512.
- Would an intact model help? Combining job-1 BASE log-probs with the emulated prior on 1024 train examples:
  best mix 0.455 vs prior alone 0.454. The base model adds nothing over the prior, so preserving it is not worth it.
- Persona-conditional priors (one demographic attribute per block, backed off): +5 pts on False consensus only,
  ~0 overall (dev 0.475 vs item prior 0.478). Response-style counts from the 8 shown facts: no gain. Dropped.
- Losses vs the exact item prior come from items sharing option tokens (econ "keep $x" items, Likert families).
  The selector could not use question tokens that also occur in options (digits), so those items shared a channel.
- v2 design (edits/r2/prior_relay.py): selector result is taken at the "Options" header token and relayed to the
  answer position (layer 1), pooling moves to layer 2, ten channels (all 12 heads of layer 0; the model is dead
  anyway). Markers: last marker token of the question decides; safe tokens first, shared tokens with repairs.
  Emulation: dev 0.4625-0.4656, held-out train 0.456 (v1: 0.455 / 0.450). Toy test exact in float32.
- Kaggle bundle limit (~1.05 MB incl. other runs' folders) hit: local-only analysis scripts moved to
  WORK/local_only/*.py.txt (not bundled); job scripts stay as .py.

## Job 3 (passes 11-20 planned): v1 from file on full dev; v2 calibration pass, v2 from file train pass + full dev
- Passes 11-14: edits/r2/prior_circuit.py loaded from file, full dev: acc 0.450, NLL 1.390 (same as in-script run).
- Pass 15 (v2 train A, same 512 indices as v1's pass A): acc 0.461 at any gain (v1: 0.451). NLL-best gain multiplier
  x16 -> stored as edits/r2/prior_relay_gain.npy (25600). Weights changed 32,258,170 / 1,543,714,304 (2.09%).
- Pass 16 (v2 from file, train B): acc 0.475, NLL 1.390 (v1 on the same indices: 0.451).
- Passes 17-20: v2 from file, full dev: acc 0.4581, NLL 1.3927. => v2 beats v1 by ~1 pt on train and dev, as emulated.

## After job 3 (offline only)
- Log-rate token tables (weights add up to a rescaled log choice rate, so gain*S would be a calibrated logit):
  emulation gives lower accuracy (holdout 0.430 / dev 0.446 vs 0.456 / 0.4625) and no better NLL. Rejected.
- Remaining gap to the exact item prior (0.478) is ~1-2 pts spread over econ "keep $x" items, small heuristics
  blocks and a few Likert items; each further fix is below the dev resolution (CI +-2.4 pts) and costs a job.

## Decision: stop. Final edit = edits/r2/prior_relay.py (+ prior_relay_{tokens,gates,markers,anchor,selector,gain}.npy)
- Full dev from file (passes 17-20): accuracy 0.4581, NLL 1.3927. Base reference: 0.294 / 2.674.
- Fallback / earlier version: edits/r2/prior_circuit.py, full dev from file 0.450 / 1.390.
- Passes used 20 of 50; Kaggle jobs 3.
- Disclosures (rule 1): the token gate tables and channel markers are per-item / per-item-group tables derived from
  TRAIN labels and written into embedding columns; channel grouping is a greedy + reassignment heuristic on those
  label statistics; kernel coefficients come from a label-free ridge least-squares fit to a ramp in token offset;
  two read-out scalars (model temperature, prior gain) were picked on train passes; design variants were compared in
  the offline emulator on held-out train people and on dev (no passes), so dev carries a little selection optimism;
  unit tests ran a RANDOM-weight Qwen2-shaped toy locally, never the real model.
- Operational note: one fetch loop exceeded the tool timeout and was moved to the background by the harness; it only
  repeated fetch.sh and no second push or fetch was started while it ran.
- Local-only analysis code lives in WORK/local_only/*.py.txt (renamed to stay under the Kaggle bundle size limit);
  run it through symlinks named *.py with PYTHONPATH pointing at them, cwd = WORK.
