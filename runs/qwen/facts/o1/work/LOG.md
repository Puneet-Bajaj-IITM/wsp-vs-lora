# facts / o1 log

## Plan (before any pass)
- Facts are about invented names; answer must come from weights. Prompt has no context.
- Hypothesis H1 (mechanism): write a "value" vector at the LAST NAME TOKEN in a mid layer, equal to the
  representation the stream has at the option token of the true value (e.g. " Lisbon"), superposed over the
  person's 3 attributes. The model's existing question/option lexical-matching circuitry should then pick
  the option whose content "appears in the question". Distractor attributes (pilot, chess) are not among
  city options, so superposition is harmless.
- Storage: repurpose MLP neurons in layer L as person detectors (key = mean MLP input at last name token,
  measured from train passes via hooks inside lab.forward), down-proj column = summed value vectors.
- Job 1: 3 capture passes (1536 train ex): keys at layers 2..16, option-token value means, random-position
  MLP inputs (for selectivity); then 6 oracle passes injecting the TRUE value vectors via hook at the name
  token to see if the value mechanism works at all, and at which layer/scale.

## Job 1 results (9 passes, 41 left)
- Base on 3x512 train: acc 0.18/0.21/0.23 (chance level), letter bias toward B.
- Keys: last-name-token MLP inputs are extremely person-separable at every layer 2..16
  (leave-one-out nearest centroid 0.99-1.00 across templates). Random-position scores on centered,
  mean-orthogonal keys: p99.99 ~0.3 vs target ~1.0. A shared direction (mean MLP input) has projection
  ~16.6+-1.1 on name tokens at L8 -> usable as bias substitute for a threshold.
- Oracle (inject TRUE value vectors = centered option-token residual mean, summed over the person's
  3 attributes, at last name token), 512 train ex:
  L4 b2 0.428 | L8 b2 0.453 | L12 b2 0.207 | L16 b2 0.188 | L8 b6 0.230 | L12 b6 0.199.
  => value mechanism works only at early/mid layers (<=8), and too large a scale destroys it.
- Numpy simulation of threshold neurons (gate=20*(s-0.6c), up=s): target act ~1.0, leakage small.

## Job 2 plan (10 passes)
- 1 targeted capture pass (least-covered persons; 1 person had no key).
- Oracles: L6 b2/b3, L8 b3/b1.5, L10 b2, L4+L8 b1.5 each, and L8 b2 asked-attribute-only (cost of superposition).
- Build mem_v1 for best oracle config (300 repurposed neurons in that layer), eval full dev (2 passes).
- Push of job 2 failed with HTTP 400 when the bundle was 3.7-11.6 MB (small bundles get the normal
  session-limit reply), so bulky arrays cannot be bundled. Job 2 now re-captures keys/values on Kaggle
  (2 passes, 3 examples per person with distinct prefixes) and builds+evaluates the edit in the same job.
  Oracle layers reduced to 6 and 8.

## Job 2 results (11 passes, 20 used, 30 left)
- Re-captured keys (900 ex = 3/person + pad, 2 passes); min 3 occurrences/person.
- Oracles (superposed 3 attributes, same 512 train set): L6 b2 0.379 | L6 b3 0.260 | L8 b3 0.336 |
  L8 b2.5 0.398 | L8 b1.5 0.549 | L6+L8 b1.25 0.426. (job1: L8 b2 0.453)
- KEY FINDING: L8 b2 with ONLY the asked attribute's value: 0.902 (NLL 0.35). Superposing the
  other two attributes' values is what costs ~45 points.
- mem_v1 = 300 repurposed neurons in L8 MLP, superposed values b1.5: DEV acc 0.508, NLL 1.966
  (city .613, prof .533, hobby .377). Real edit ~ oracle (keys are fine); the value superposition is the bottleneck.
- Next: make the written value attribute-specific. Options: (a) orthogonalise each attribute's
  value subspace against the others; (b) gate (person, attribute) neurons on an attribute direction
  present at the name token (all dev/test templates cue the attribute BEFORE the name).
- Local analysis: attribute is readable at the last name token at L8 when the question names it
  before the name (in-sample asked score 1.0+-0.5 vs other -0.5), but transfer to a held-out
  template is weak (asked 0.3-0.4, argmax 0.67). Value subspaces of different attributes are nearly
  orthogonal (mean |cos| 0.05), so linear crosstalk is not the main superposition cost; the total
  injected norm (3 x ~32 vs residual ~45) probably is.

## Job 3 plan (13 passes)
- 2 capture (same deterministic selection as job 2), 7 oracles: superposed b1.0/b0.75; asked-only
  b1.5/b3; asked b2 + others as random equal-norm vectors (norm vs content test); asked b2 + others b0.5/b1.
- Build mem_g1: 900 neurons (person x attribute), written weight c0 + gamma*m_a(x) with
  c0 = best superposed beta, c0+gamma = best asked beta. Build mem_v2 (superposed at best beta) if != 1.5.
- Dev eval of both (2 passes each).
- Job 3 attempt 1 crashed (cpu/cuda mix in random-norm oracle) after its 2 capture passes (22 used). Fixed; rerun.
- Job 3 attempt 2: 2 capture + 7 oracles ran, then crashed at a dict(**float keys) bug before
  building edits (31 passes used, 19 left). Oracle results (L8, 512 train):
  sup b1.0 0.580 | sup b0.75 0.547 | asked-only b1.5 0.930 | asked-only b3 0.822 |
  asked b2 + others RANDOM equal-norm 0.502 | asked b2 + others b0.5 0.885 | asked b2 + others b1 0.789.
  => The superposition cost is NOT content crosstalk: random vectors of the same norm hurt as much.
     It is the extra norm written at the single name position (disrupting it).
- New hypothesis H2: write each attribute at a DIFFERENT name token (last / penultimate / antepenultimate),
  so no position carries more than one value's norm.

## Job 4 plan (7 passes; 31 used before)
- 2 capture passes, now also MLP inputs at the 2nd/3rd-from-last name tokens (L8).
- 1 oracle: spread layout city@last, profession@-2, hobby@-3 name token, b1.5 each, true values.
- mem_s1: 900 neurons at L8, one per (person, attribute), each keyed on the person's representation
  at its assigned name token (fallback to last token if that key collides with another person, cos>0.8).
- mem_g1: 900 neurons at L8, all keyed at the last token, weight 0.75 + 1.0*m_a(x) (attribute score).
- Dev eval both (4 passes).

## Job 4 results (7 passes, 38 used, 12 left)
- Name lengths: 4-8 tokens (mostly 5-6), so the last 3 name tokens are surname pieces.
- Spread oracle (city@-1, prof@-2, hobby@-3, b1.5, true values): 0.889 (city .957 prof .937 hobby .778).
- mem_s1 (spread, 900 neurons L8): DEV 0.713, NLL 1.442 (city .907, prof .770, hobby .463).
  Key collisions (cos>0.8 with another person) fell back to last token: offset2 25, offset3 61.
- mem_g1 (gated, all at last token): DEV 0.654, NLL 1.394 (city .897, prof .523, hobby .543).
- Local diagnosis of mem_s1 on dev: city 0.982 for persons with no fallback but 0.684 when another
  attribute fell back onto the last token (norm cost again). Keys at offset 3 are poorly unique
  (61 persons with cos>0.8 to another; best-other p90 0.98); hobby at its own offset-3 neuron only 0.448
  on dev (train oracle 0.78).
## Job 5 plan (6 passes -> 44 used): recapture; mem_s2 = city@-1, hobby@-2, prof@-3 (b1.5);
   mem_s3 = mem_s1 layout with hobby beta 2.0. Dev eval both.

## Job 5 results (6 passes, 44 used, 6 left)
- mem_s2 (city@-1, hobby@-2, profession@-3, b1.5): DEV 0.730, NLL 1.163 (city .900, prof .663, hobby .627).
- mem_s3 (mem_s1 layout, hobby b2.0): DEV 0.690, NLL 1.534 (hobby .423) -> larger beta does not rescue offset -3.

## Decision: stop. Final edit = edits/facts/o1/mem_s2.py (+ mem_s2_mem.pt, _memlib.py).
- Remaining ideas (rerouting offset-3 collision fallbacks to offset 2 instead of the last token) are
  worth ~1-2 points at most, inside the dev CI, and would cost 4 of the last 6 passes plus another round
  of selecting on dev. Not worth it.
- Weights changed: 900 neurons of layer-8 MLP (gate row, up row, down column each) = 4,147,200 params
  = 10% of L8 MLP, 0.27% of the model.
- Rule-1 note: the down-projection columns are a per-person table derived from train labels (each
  person's true city/profession/hobby -> the centered option-token mean of that value). No fitting/solver.
