# sst5 / r3 log

## Setup (local, free)
- Class frequencies train: vneg .128, neg .260, neu .190, pos .272, vpos .151 (dev similar).
- Base reference: dev acc 0.423, NLL 3.436 -> heavily overconfident; accuracy low for a model that "knows" sentiment.
- Hypotheses:
  H1 class-level bias (e.g. over/under use of "very ..."/"neutral") costs accuracy; options are shuffled, so a fix
     must act on option content, not on letters.
  H2 letter bias + overconfidence: fixable at the readout (letter rows of the tied embedding, final norm gain).
  H3 the final residual may linearly encode "class c sits at letter l" (prompt structure); if so a class-prior
     correction can be written into the 5 letter rows in closed form.

## Job 1 (12 passes planned)
- 4 base passes on train[0:2048] capturing residual at layers 6,10,14,18,21,24,26,27,-1.
- Probes on train[0:512]: scale embedding row of " very" / " neutral" (x0.6, x1.5); scale attention o_proj of
  layers 20-27 and 10-19 (x0.8, x1.25).

### Job 1 results (12 passes used)
- Base train[0:2048]: acc 0.435, NLL ~3.16. Polarity is essentially never wrong. The errors are intensity errors:
  "neutral" is predicted for 30/2048 items (true share 19%), true-neutral goes mostly to "negative";
  "negative" is over-read as "very negative", "positive" as "very positive".
- Scaling the " very" / " neutral" embedding rows (x0.6, x1.5): no effect at all (H1 via option tokens: dead).
- Attention o_proj scale: late x1.25 0.463, mid x1.25 0.441, x0.8 0.42-0.44 (base 0.428 on the block): within noise.
- Offline (an1.py): an oracle per-class logit bias (vneg -4, neu +3.5, vpos -3) gives 0.50 on held-out half, so
  class calibration is worth about +7 points. H3 fails: option placement is only weakly decodable at the answer
  position (best at layer 14, ~0.5), so the bias cannot be written into the letter rows.
- Offline (an2.py): at the answer position the review's sentiment is absent at layer 14 and present at layer 18
  (class-centroid decode 0.50). A "mild minus extreme" mean-difference direction exists; neutral reviews sit on the
  negative side of the polarity axis, which explains neutral -> negative.

## Job 2 plan (22 passes)
- Mechanism: add a constant vector along the "mild" direction to the residual stream by editing attention
  v_proj biases (softmax rows sum to 1, so the bias passes through o_proj to every position). GQA: only 256 bias
  dims per layer, so 4 consecutive layers are solved jointly (pinv on the weights) to reach the target vector.
- Grid on train block 0: families A (layers 16-19, direction from L18), B (19-22, L21), C (22-25, L24);
  directions mild / mild orthogonal to polarity / neutral orthogonal to polarity; several strengths.
  Then top 3 on block 1, pick best two-block mean, pick logit temperature (final norm gain) on the pooled train
  logp grid, write the edit, measure full dev.

### Job 2 results (22 passes, 34 used in total)
- v_proj-bias injection over 4 layers reaches only ~64% of the target vector (1024 of 1536 dims).
- Family A (layers 16-19, direction from layer 18) moves the class histogram as intended: neutral predictions
  11 -> 38 (x2) -> 100 (x4) -> 170 (x8) of 512; accuracy peaks at x2 (0.46 vs 0.43 base on block 0), collapses at x8.
  Families B/C (later layers) barely change the class histogram: the mild/neutral decision is made by ~layer 19.
- Side effect: steering creates a strong letter bias (B over-predicted, D/E under-predicted).
- Selected A/mildperp x2 (two-block mean 0.463), logit scale 0.26. Edit `steer_mild`:
  **dev acc 0.4478, NLL 1.3029** (base 0.423 / 3.436).
- Offline (an3.py): a label-free letter debias (offsets that make the mean predicted letter distribution uniform)
  on top of A/mildperp x2 gives 0.492 on block 0 and 0.482 on block 1 (offsets from block 0). The mean final hidden
  state mu is a near-constant direction (h.mu/|mu|^2 has sd 3%), so offsets can be written into the A-E rows as
  b_l * mu/|mu|^2. Mid-layer o_proj x1.25 also looks better after debias (0.475 vs 0.443 on block 0).

## Job 3 plan (12 passes)
- Explore on train block 2 (capture final hidden): A4 x2, 6-layer injection (14-19) x2 / x3, layers 17-20 x2,
  A4 x2 and x1.4 combined with o_proj x1.25 on layers 10-19. Score = accuracy after label-free letter debias.
- Top 2 again on block 3; choose by pooled score; write edit `steer_mild_debias` (v biases + letter rows +
  final-norm gain); verify the file on block 1; full dev.
