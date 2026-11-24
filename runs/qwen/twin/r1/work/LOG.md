# WSP log: Qwen2.5-1.5B-Instruct -> Twin-2K respondent prediction

## 0. Free local analysis (no passes; train/dev jsonl + results/logp_base_dev.npy)
- Dev is 38.5% Personality (Likert-5 items dominate), 33.6% Cognitive, 12% False consensus (policy
  support/oppose), 7.6% Economic (dictator splits), ~8% small judgment blocks.
- Base model on Likert personality (529 dev ex): acc 0.21. It picks "Agree strongly" 46% of the time;
  people pick it 12% (people: "Agree a little" 30%, "Disagree a little" 21%, "Neither" 19%).
  Best deterministic remap of (model pick text -> predicted text) reaches only 0.306 ~= "always
  Agree a little" (0.299). Model's direction (agree vs disagree) agrees with people 60% per example;
  item-level agree-mass correlates 0.655 with population agree rate (so some item knowledge exists).
- False consensus (194 dev): base picks "Strongly oppose" 42% of the time, people 9%; people mostly
  "Strongly/Somewhat support". Best remap 0.39; "always Strongly support" 0.356.
- Position bias is block dependent: letter A is picked 67% on Economic items, 1% on False consensus.
- Copying a same-scale Likert answer from the 8 context facts: only 0.298 -> no gain over the mode.
- Conclusion: realistic levers are (1) temperature (NLL), (2) letter-position offsets,
  (3) option-text preference (population mode), (4) a difference-of-means readout from the final
  hidden state into the letter unembedding rows. Beating the per-item prior (0.478) would require an
  item-level table; not attempted first.

## Mechanisms planned (all closed form)
- Temperature: scale `model.norm.weight` by alpha (scales every logit -> exact softmax temperature).
- Constant letter offsets: add delta to layer-27 `v_proj.bias`; since attention weights sum to 1 this
  injects `W_o @ tile(delta)` into every residual position -> constant shift of letter logits through
  the final norm. Solve 5 equations in 256 unknowns (min-norm) for desired offsets.
- Readout: mean(h_final | label=X) - mean(h_final) added to W_U[letter X] (tied embeddings: flag).
- Lexical remap: swap embeddings of ' strongly' <-> ' little' (probe only).

## Job 1 (6 passes, 44 left): base captures 5x512 train, lexswap_v1 1x512 train
- Captured final-norm hidden state reproduces the letter log-probs exactly (max diff 0.023, fp16), so
  edits to the final norm scale / letter unembedding rows can be simulated offline at zero pass cost.
- |h_final| ~ 162, letter embedding rows ~1.0 (tied), embeddings tied (confirmed).
- Base on 2560 train: acc 0.313, NLL 2.57. Temperature alpha=0.3 -> NLL 1.55 (acc unchanged).
  Letter offsets (cancel mean letter logit) -> +0.3% acc only: the shuffle makes position bias
  label-neutral, so it costs NLL but little accuracy.
- Difference-of-means readout (mean(h|label=X) - mean(h)) into W_U letter rows, 2-fold CV by person:
  best +1.2% acc (0.316 -> 0.328) with beta=0.01, PCA-64, and NLL explodes. Class-mean separation is
  ~5 against noise in 1536 dims; the final hidden state does not linearly encode "the popular option is
  at letter X" well enough. DROPPED as a lever.
- lexswap_v1 (' strongly'<->' little'): paired on the same 512 train examples, Personality 0.215 -> 0.294,
  overall 0.330 -> 0.359, NLL 2.55 -> 2.48. Picks moved from "Agree strongly" to "Agree a little".
  This is the strong lever: route the model's existing option preference onto the population mode.
- Other free findings: on cube-rotation items (option texts are letters) the model picks the option whose
  text is "A" 55% of the time it is shown (true rate 18%, "D" 33%); "Mr. A 1" picked 23/24 (truth ~0);
  "Often" picked 74% on the Never..Always scale (truth mode "Rarely"). Policy items: everyone mostly
  supports (Dem 76%, Ind 65%, Rep 53%) while the model predicts Democrats oppose 53% -> its stance is noise.
- Mixing base logp with the population prior never beats the prior alone -> the base model carries no
  information complementary to item frequencies; without an item table the ceiling is ~0.43 (text-level
  popularity rule) and realistically ~0.35-0.38 via lexical routing.

## Job 2 (6 passes): block-targeted paired probes
- FC-targeted 512: base vs lex_fc_weak (' oppose' := ' support' + 0.5(' little'-' strongly')) vs
  lex_fc_neutral (' oppose' := ' neutral').
- Cube/Mr-targeted 512: base vs lex_cube_v1 (' A' <-> ' D').
- Personality-targeted 512: lex_pers_v1 (strongly<->little, Often<->Rarely), unpaired vs the 950
  Personality examples in the job-1 base captures (0.212).

### Job 2 results (12 passes used, 38 left) -- paired on the same 512 targeted train examples
- FC: base 0.238 | lex_fc_weak 0.217 (FAILED: "Strongly oppose" still picked 160x; the weakened-support
  reading did not break ties toward the real option) | lex_fc_neutral 0.289 (picks "Strongly support"
  55%, residual "Strongly neutral" picks 10%). Keep ' oppose' := ' neutral'.
- Cube/Mr: base 0.160 | ' A'<->' D' swap 0.225 (cube subset 0.166 -> 0.240; picks of text "D" 33 -> 131).
- Personality: lex_pers_v1 0.272 (Likert subset 0.276 n=457; Never..Always subset 0.37 n=27) vs base 0.212.
  ' Rarely' is two tokens (' Rare','ly') so that swap was skipped -> use ' Often'<->' Sometimes' instead
  (' Sometimes' single token; 33% of real answers vs 37% for Rarely, 19% for Often).
- After the swap the model still says "agree" 73% of the time (truth 44%); its per-example direction is
  only 60% right, so no further Likert routing is justified without item knowledge.

## Job 3 (5 passes, 17 used / 33 left): composite `lexroute_v1`
- Remaps: strongly<->little, Often<->Sometimes, ' oppose':=' neutral', ' A'<->' D'.
- Phase A (1 fresh train pass, alpha=1): acc 0.334 (base 0.313 on the job-1 sample), NLL 2.61.
  alpha* = 0.175 (train NLL 2.61 -> 1.51). Letter offsets lowered train acc (0.334 -> 0.326) so they
  were NOT written (edit = remaps + temperature only).
- Phase B, FULL DEV (1,600 ex, 4 passes): **acc 0.3319 (CI 0.309-0.356), NLL 1.525**
  vs base 0.2938 / 2.674 (paired: 167 wrong->right, 106 right->wrong).
  By block: Personality 0.188 -> 0.273 (Likert subset 0.214 -> 0.288, Never..Always 0.085 -> 0.277),
  Cognitive 0.460 -> 0.471 (cube items 0.10 -> 0.24; antonyms unchanged), False consensus 0.258 -> 0.253,
  Economic 0.223 -> 0.231, small blocks mixed (n<15 each).
- Policy block per item: the four "A ..." items improved (so no harmful interaction with the ' A'<->' D'
  swap); "Increasing deportations" and "healthcare vouchers" dropped because real respondents do oppose
  those and the edit removes the model's ability to say so. Net wash on dev (+5 pts on the train probe).
- Residual inefficiencies: 149 Likert dev picks still land on the real extremes (the model's *mild*
  preference lands on them after the swap); 42 policy picks still land on the "neutral"-reading oppose
  options. Both motivate mapping those words to a non-answer instead of another scale point.
- Weights changed: 7 embedding rows (6 swapped + ' oppose') x 1536 + final norm (1536) = 12,288 of
  1,543,714,304 (8.0e-6).
- Housekeeping: fetch.sh wipes kaggle_runner/out/wsp each job, so the job-1 hidden captures and job-2
  logp files are gone locally (all their results are recorded above); job-3 outputs archived under
  kaggle_runner/out/archive/job3.

## Job 4 (3 probe passes, + 5 if a variant wins): `lexroute_v2`
- Likert probes on the job-2 Personality indices (paired vs swap 0.276 on the Likert subset):
  v2a ' little':=orig ' strongly', ' strongly':=' neutral';  v2b same but ' strongly':=meaningless token.
- Policy probe on the job-2 FC indices (paired vs ' neutral' 0.289): ' oppose':=meaningless token.
- Auto-select per block with a 0.01 margin; if nothing wins, lexroute_v1 stands and no dev passes are
  spent. Otherwise: fresh train pass for alpha, then full dev (4 passes).

### Job 4 results (3 passes, 20 used / 30 left) -- paired on the job-2 targeted indices
- Likert subset (n=457): swap 0.276 | v2a (' strongly':=' neutral') 0.263 | v2b (' strongly':=' xyz')
  0.221. v2b FAILED badly: the model reads "Agree xyz" as a bare "Agree" and picks the real extreme
  170/512 times. Mapping to a non-answer is worse than the swap -> keep the swap.
- Policy (n=512): ' neutral' 0.289 | ' xyz' 0.297 -> within the 0.01 margin, keep ' neutral'.
- Selection: no change. **lexroute_v1 is the final edit**; its full-dev numbers (job 3) stand.

## Final summary
- Edit: `edits/lexroute_v1.py` + `edits/lexroute_v1_alpha.npy` (alpha = 0.175).
- Full dev (1,600 ex, 4 passes): acc 0.3319 (95% CI 0.309-0.356), NLL 1.525.
  Base 0.2938 / 2.674; population prior 0.478 / 1.286. Not beating the prior.
- Passes: 20 of 50 (job1 6, job2 6, job3 5, job4 3). Kaggle jobs: 4.
- Weights changed: 12,288 of 1,543,714,304 (8.0e-6): 7 input-embedding rows (tied with lm_head) and the
  final RMSNorm scale vector.
- Failed / dropped: ' oppose' := weakened ' support' (-2 pts), ' strongly' := ' neutral' (-1.3 pts),
  ' strongly' := meaningless token (-5.5 pts), difference-of-means readout into W_U rows (+1.2 pts CV,
  NLL explodes), constant letter offsets (-0.8 pts on the train pass). Economic-block letter-A bias and
  cognitive antonym errors left untouched (no closed-form lever found).
