# twin / r4 log

## Local data analysis (free)
- train 16800 ex, dev 1600; 92 items, 230 distinct option strings, 533 (item, option) pairs.
- Dev accuracy of label-count priors: per-item 0.479, global option-text 0.426. By block (item prior):
  Cognitive tests 0.635, Economic 0.537, Personality 0.375, False consensus 0.361.
- 76% of examples are 5-option items (all options always shown, shuffled).
- Hypothesis: base (0.294) lacks the population prior; the win is to get "which option content is popular
  for this question" into the answer computation. Options are shuffled, so a pure answer-position letter
  readout cannot carry a per-item prior; the lever must act on option content (key side) or on the
  model's content guess (query side).

## Job 1 (step1.py): 4 instrumented base passes, 2048 stratified train examples
Read-only hooks during lab.forward: answer-position attention per head onto segments (options split in
letter/content/end), pre-RoPE q.k match per option, per-item MLP neuron means at answer position,
residuals at option last/end tokens (chosen-vs-not directions), answer-position residuals at 12 layers.

### Job 1 results (passes 1-4, base, train)
- Base on 2048 stratified train: acc 0.309, NLL 2.556, letter hist A.14 B.21 C.28 D.25 E.12. Blocks: Cognitive 0.49,
  Personality 0.195 (picks "Agree strongly" 46% of the time; labels favour "Agree a little"/"Disagree a little"),
  False consensus 0.255, Economic 0.237.
- Where the letter is decided: answer-position residual knows the predicted letter only from L19-21 on
  (nearest-centroid 0.31 @L17, 0.36 @L19, 0.75 @L21, 0.90 @L23). No linear "content guess" earlier.
- Attention from the answer position: L18H3, L20H{2,4,5,9,11}, L21H{8,9,10,11} read option content/end tokens and their
  argmax option agrees with the final prediction 50-69%; then L22H7 (0.93 mass on option letter tokens, 71% agreement)
  and L23H8 copy the letter.
- The answer-position query of those heads is a FIXED vector (per-item mean queries have cos 0.97-0.99 with the global
  mean; >=94% of its energy in low-frequency RoPE dims). The pair-mean key score q̄·k at an option's end token
  correlates 0.75-0.82 (within item) with how often the model picks that option, but only ~0.3 with the true
  log-prior. => the model's choice is a static content "attractiveness" of each option, read by a fixed query.
- "Answer-ness" directions in the residual at option tokens are weak (split-half 0.28-0.35), label version ~chance.
- Read-out of the prior table in the 256-d key space is poor (keys nearly rank-16): Hebbian corr 0.45.

### Plan (job 2)
Key-side edit "KQ": k_proj[L] += beta * u_L w_L^T for L in 18,20,21: w_L = read-out vector in the k_proj input space
whose value at an option's last/end token is the option's centred log-prior (table from train counts);
u_L = sparse low-frequency key direction with q̄_h·u = sqrt(d) for the option-reading heads. Capture the needed
activations (4 passes), measure held-out retrieval for several read-out constructions, then test edits (1 pass each)
on job-1 pass-0 examples (base logp known, paired).

### Job 2 results (passes 5-14)
- Capture (4 base passes, 2048 new train ex). Held-out per-instance retrieval of the centred log-prior table by a
  linear read-out of the k_proj input at an option's last/end token (corr with table / argmax==table argmax):
  L18: hebb .45/.49, hebb_z .49/.49, rome(C^-1 generic) .55/.52, pcaw128 .50/.43; L20, L21 similar (rome .52-.54).
  MLP-activation space (L13/15/17, Hebbian): .39-.51 -> no better. Retrieval is the bottleneck (exact table = .478 acc,
  read-out argmax vs label only .33-.38).
- u vectors: 26-53 key dims per layer give gain 1.0 on selected heads (and 0.5-1.0 on the other heads of the kv group).
- Edit tests on job-1 pass-0 (512 train ex, base acc .287, NLL 2.55):
  rome b1.5 .334 | rome b3 .320 | rome b6 .264 | hebb_z b3 .338 | pcaw b3 .3125 | hebb_z b6 .270.
  The lever is causal (36-69% of predictions change; Personality .17->.25-.28, False consensus .30->.37) but large
  beta hurts and pushes predictions to letter E (hist E .34): the read-out has slot-specific offsets (E's end token
  is "\n\n", tokens also encode their slot), i.e. the edit injects letter bias + retrieval noise.
- Decision: (1) remove the slot-mean subspace from w and centre keys per example; (2) save raw option-token inputs
  so read-outs can be designed offline; (3) keep MLP-space diagnostics at more layers.

### Job 3 FAILED after its 6 capture passes (passes 15-20 wasted)
step3.py read j2/consts_rome.npz after the capture, but the bundler ships only .py/.md/.json/.npy/.pt, so the file
was missing on Kaggle and the script died before saving anything. 20 passes used, 30 left.
Fix: constants converted to .npy, every input is loaded before the first pass, raw captures are saved before analysis.
Re-run with 4 capture passes.

### Job 3 rerun (passes 21-28): 4 capture passes (raw L18/20/21 option-token k_proj inputs saved) + 4 tests
- v2 read-outs (slot means removed at last/end tokens, per-example centring): held-out retrieval unchanged
  (rome corr .50-.55, argmax==table .47-.53, argmax==label .32-.35).
- Tests (same 512 eval ex, base .287): rome b2 .311 | rome b4 .271 | rome b8 .236 (hist E .475) | hebb_z b4 .322.
  So the slot fix at last/end tokens did NOT remove the E bias; large beta still destroys accuracy.
- Offline analysis of the raw captures (a6-a9, b1):
  * Read-out vs its own pair mean correlates .92 -> instance noise is not the bottleneck; the Hebbian read-out only
    reproduces the table with corr .63 even on the pair means it was built from (crosstalk between overlapping keys).
  * Joint (decorrelated, regression-like) solves were tried as a DIAGNOSTIC only: no better on held-out examples
    (label acc .36-.38 vs .354) -> not used.
  * Stand-alone label accuracy of the read-out argmax is ~.35 (exact item table .457, exact text table .405):
    the linear read-out caps this mechanism around .35.
  * The read-out responds to generic tokens with rms ~0.65-0.75 of its option-token signal sd: at beta>=4 hundreds of
    non-option tokens (facts contain the same answer strings; prefix sink; tail) get large key boosts = collateral.
- Decision (job 4): project the read-out off (a) slot-mean differences of every option token type (letter, dot,
  first/last content, end, content mean), (b) the constant prefix vectors, (c) tail position means; optionally add a
  uniform "is an option token" gate (gamma); record attention segment masses in every test pass.

### Job 4 (passes 29-34): nuisance capture + KQ v3 tests with attention probe
- Nuisance projection (66 directions: slot differences of 6 token types, 24 prefix vectors, 18 tail means) keeps 98%
  of the read-out norm; the "is an option" gate separates cleanly (options +0.4..0.8, generic -0.25, sd 0.09).
- Tests (eval1, base .287): b2 g0 .318 | b4 g0 .299 | b4 g.5 .273 | b8 g.5 .248 | b2 g1 .318. Gate does not help.
- Attention probe: under the edit the heads look at options (content+end 0.7-0.9 of mass), and the option they prefer
  matches the table argmax ~50% at every beta (the read-out's ceiling; base 37-39%). At beta 2 the final prediction
  follows the edited heads (agreement .69-.85; pred==table .46 vs .39 base). At beta>=4 attention becomes very peaked
  (top option share .76-.88) and the downstream letter-copy stage (L22H7/L23H8) STOPS following (agreement .33-.38 at
  beta 8, L22H7 mass on letters falls .81->.55, tail .11->.19; predictions drift to E). So: collateral is not the
  problem; over-sharpening breaks the downstream read, and retrieval quality caps the gain. Moderate beta only.
- Offline (b2-b5): a "chosen minus not-chosen" difference of means over option tokens (label indicator and/or the
  item's train answer frequency as weights, whitened by the generic second moment) retrieves better than the
  per-pair Hebbian table write: stand-alone held-out label accuracy .39-.40 vs .354. Conditional (per-item) read-outs
  would reach .435 with oracle item gating but need 92 gated components -> not attempted.
- Decision (job 5): KQ-ld = same k_proj edit with the difference-of-means read-out; test beta 1.5/2.5/4 on eval1, then
  the best and the job-2 'rome' beta 1.5 variant on eval2.

### Job 5 (passes 35-39): KQ-ld tests
- ld (label+frequency difference of means) eval1: b1.5 .305 | b2.5 .309 | b4 .270; eval2 b2.5 .320 (base .328).
- job-2 'rome' b1.5 on eval2: .354 (base .328). Pooled over eval1+eval2 (1024 ex): rome b1.5 .344 vs base .308.
- Why ld is worse despite better stand-alone retrieval (b6/b7): its read-out correlates .68 with the base model's own
  preference (a "model-probability" difference of means reproduces the base preference with corr .85, i.e. the
  native attractiveness is itself linear in the k_proj input). ld therefore mostly sharpens what the heads already
  prefer, and over-sharpening breaks the downstream read. The table read-out ('rome') correlates only .3.
- Offline proxy (acc of base_logp + c*read-out, cross-fitted): subtracting a share m of the model-probability
  weights ("target minus model") adds information while flattening the native preference: m=.25 -> .37-.38 at c=2-3,
  vs .34-.36 for m=0 and .33 for the table read-out.
- Decision (job 6): test m=.25 (b 2, 3.5) and m=.5 (b 2, 4) on eval1, best on eval2, pick it only if its pooled
  accuracy beats rome b1.5 (.344); temperature tau = value whose mean confidence equals measured accuracy on the
  pooled train evals; write constants, reload through edits/r4/kq_final.py, run full dev (4 passes).

### Job 6 (passes 40-48): target-minus-model read-outs, selection, final dev
- eval1 (base .287): m25 b2 .332 | m50 b2 .326 | m50 b4 .270 | m25 b3.5 .324. eval2 (base .328): m25 b2 .369.
- Pooled 1024 train ex: m25 b2 .3506 vs rome b1.5 .3438 vs base .3076 -> m25 b2 selected.
- tau = 0.2 (mean confidence == accuracy on the pooled train evals).
- FINAL edits/r4/kq_final (m25, beta 2.0, tau 0.2), loaded through the edit file, full dev (1600):
  acc 0.35625 (bootstrap 95% CI .333-.381), NLL 1.5034, letter hist A.27 B.19 C.19 D.23 E.11.
  Blocks: Cognitive .518, Personality .235, False consensus .330, Economic .380, other .280.
  Reference base dev: .294 / NLL 2.674 (not re-measured in this run).
- 48 of 50 passes used (44 train, 4 dev), 7 Kaggle kernel versions (one failed after its capture passes).
- Weights changed: 111 key rows x 1536 in k_proj of L18/20/21 (170,496) + 1,536 final-norm gains = 172,032 (1.1e-4).
- Stopped: 2 passes left cannot fund another full-dev measurement.
