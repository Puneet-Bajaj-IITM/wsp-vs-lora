# facts / r2 log

## Local analysis (free)
- 900 facts (300 people x 3 attributes), prompt carries no context. Train templates 0-2, dev 3, test 4.
- Only the answer position can be captured, and the base model cannot know the facts, so there is no
  natural "answer" direction to amplify. The fact table has to be written into weights as a lookup.
- The letter depends on the random option order, so a lookup alone is not enough: the circuit must
  bind (person -> value) with (value -> slot letter) per prompt.

## Hypothesis H1: fully analytic circuit in layers 0-1
Layer 0 input is exactly RMSNorm(token embedding), so attention there can be designed in closed form
from embedding rows, with no activation statistics at all.
- Head N (layer 0): from "\n" queries attend equally to name tokens + the 3 `<|im_start|>` tokens
  (fixed-count sink, so the bag is un-normalised). OV writes one orthogonal code per name token.
- Head A (layer 0): from the first token of an option value attend to the letter 2 tokens back
  (RoPE kernel peaked at offset 2). OV writes a letter code.
- Head I (layer 0): the option value token attends to itself and writes a value-identity code.
- MLP 0: 300 repurposed neurons, one per person. Gate = linear threshold on the name-token bag
  (integer margin), down = sum of the codes of that person's three true values. This is the
  per-entity label-derived table written into the weights.
- Head B (layer 1): query = person's value codes at the final position, key = value-identity code at
  option tokens, OV maps the letter code to a large write along the letter unembedding direction.
- Later layers are left alone; the write is made large enough to dominate the answer position.

## Job 1 (dump + 1 base pass)
Dump tokenisation, used embedding rows, layer 0/1 norm gains and V projections, final norm gain.
One base pass on dev[0:512] with all-layer capture for residual norm scales.

## Revision before any pass (local tokenizer analysis, free)
- Dump job dropped: the edit builds itself from the weights inside `apply()`, so no weight download is needed.
- Names tokenise into 267 distinct pieces, more than a 128-dim head value space can map exactly.
  Greedy selection keeps 94 name tokens (none repeated inside a name, none shared with the prompt frame
  or option values) such that every pair of people differs in >= 2 kept tokens. The person neuron's
  gate is sum_t b_t(2 b_p,t - 1) - (F_p - 1): +1 for the right person, <= -1 for everyone else.
- Option values are multi-token and " H" / " T" collide with name pieces, so each value gets one
  distinctive token and its own offset (2-3) back to the letter.
- Logic check on a RANDOM-INIT 3-layer Qwen2-shaped mock (not the study model, local CPU, no ledger):
  96/96 correct in float32 and with fp16-rounded weights; bag read-out error 0.04, value-code 0.99.
- Kaggle push blocked repeatedly by the 2-GPU-session limit (other agents).

## Job 1 plan (5 passes)
Train probe (512 fixed random train items, capture layers 0,1,2,5,12,20,27,-1) for three scale variants:
hijack_v1 (codes 300, write 4000), hijack_w16k (write 16000), hijack_c1k (codes 1000, write 16000).
Best by train-probe accuracy gets a full dev measurement (2 passes).

## Job 1 results (kernel version 1 died on an import-order bug before any pass; version 2 ran)
Real-model design checks printed by apply(): head value-space condition numbers 51 / 3.1 / 9.8 / 87,
min bag margin 2, 3,705,795 of 1,543,714,304 weights changed (0.24%).
Train probe (512 items, 1 pass each):
- hijack_v1   acc 1.000, NLL 0.0118; layer-0 value-code read-out 100%, bag error 0.026.
- hijack_w16k acc 1.000, NLL 0.0095; letter read-out gap 5.98 after layer 1, 5.34 at the final norm.
- hijack_c1k  acc 1.000, NLL 0.0103.
- hijack_late (selection head in layer 26) acc 1.000, NLL 0.0145: codes survive 26 natural layers.
The 26 untouched layers shave only ~0.6 nats off the designed 6-nat gap, so the write dominates as intended.
Full dev (2 passes) for hijack_w16k: acc 1.000 (900/900), NLL 0.00954, letter hist [.200,.213,.196,.192,.199].

## Decision
Stop at 6 of 50 passes. Dev (unseen template 3) is perfect because the circuit keys on name tokens and
option tokens only, never on the question wording. Residual risk for test: a template-4 word that happens
to be one of the 94 kept name tokens would add a spurious token to the bag and block the person neuron.
Not checked (test split is off limits). Logit gap left at 6 nats so that any such miss costs ~6 nats, not more.
Final edit: edits/facts/r2/hijack_w16k.py.
