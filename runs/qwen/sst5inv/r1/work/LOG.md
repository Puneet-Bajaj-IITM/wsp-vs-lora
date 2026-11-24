# sst5inv r1 log

## Local analysis (no passes)
- Prompts are identical to SST-5; label = mirrored sentiment. Options are the 5 fixed strings, shuffled.
- Label words appear inside review text in only 10/8544 train and 0/1101 dev reviews.
- H1: the base model reads sentiment correctly and picks the true-sentiment option, so on the inverted labels it
  is right mostly on neutral items only. Swapping the embedding rows of " negative" and " positive" makes every
  option line read as its mirror, turning the task back into plain SST-5 for the model.
- H2 (after H1): remaining error is the model's own SST-5 confusion (very vs plain, neutral); look at the
  confusion matrix and decide whether class calibration is worth it.

## Job 1 plan (5 passes)
base train[0:512] with captures; base full dev (3 passes); swap_polarity train[0:512] with captures.

## Job 1 results (5 passes used, 45 left)
- Tokens: " negative"=8225, " positive"=6785, " neutral"=20628, " very"=1602; options are "X", ".", " word"; embeddings tied.
- Base full dev: acc 0.0500, NLL 9.396 (n=1101). Base reads the true sentiment: read as plain SST-5 it scores 0.425 on dev,
  0.490 on train[0:512]. So H1 holds: knowledge present, readout mirrored.
- swap_polarity on train[0:512]: acc 0.412, NLL 3.23. Same distribution as base's SST-5 reading (0.43-0.49); gap to 0.49 on
  the same reviews is option-order noise (dev base-read is 0.425).
- Model's SST-5 confusion: predicts neutral ~2% of the time (true 19%), over-predicts "very" classes.
- Offline (no passes) class-offset sweep on logged letter log-probs: very -3..-5 and neutral +2..+4 are worth about
  +3 to +5 points; letter-position offsets are worth nothing. Logits are overconfident (NLL 3.2 > chance 1.61).
- Deleted hidden-state arrays after analysis to keep the shared Kaggle bundle under its size cap.

## Job 2 plan (11 passes)
- H3: making the "very" options read as more extreme (" very" row <- intensifier donor) lowers very-picks.
- H4: making the neutral option read as "mixed"/"average" raises neutral picks.
- H5: steering the answer-position residual against the model's own "very" direction (mean h | predicted very minus
  mean h | predicted plain) at layer 16/19, and toward the true-neutral direction, via rank-one down_proj updates keyed
  on the mean answer-position MLP activation. Keys captured with forward pre-hooks that run inside the budgeted
  lab.forward pass (flag as borderline in report).

## Job 2 results (11 passes, 34 left) - all on train[0:512], swap baseline 0.412 / NLL 3.23
- Sentiment reaches the answer position at layers 15-16 (valence diff norm 0.2 at L12, 0.95 at L15, 7.3 at L16, 14 at L19).
- Relabel variants: very<-extremely 0.428; very<-half blend of 6 intensifiers 0.4375; neutral<-mixed 0.398 (worse, neutral
  picked even less); neutral<-half blend 0.418. H4 rejected; H3 weak (+1.5..2.5, within noise).
- Steering (rank-one down_proj, key = mean answer-position MLP activation restricted to top 512 neurons; gain at answer
  position 1.00 +- 0.03..0.05, leak at an option-area position 0.06): -intensity at L16 x1 0.422, x2.5 0.432;
  at L19 x1 0.434, x2.5 0.445 (NLL 2.57, very-picks 196 -> 152, true 147). Neutral steer at L16 x1/x2.5 0.414/0.418 (too weak;
  direction norm only 1.6 there). H5 supported in direction; effect matches the offline "very -2" offset (0.443).
- neutral and intensity directions are anti-aligned (cos -0.69 at L19), so -intensity also raises neutral picks (12 -> 25).

## Job 3 plan (9 passes)
Stronger steer at L19/L20 (intensity x4, x6; L20 x2.5, x4), neutral at L19 (x2.5, x5), combo; auto-pick best on train A,
then compare with swap baseline on fresh train[512:1024].

## Job 3 results (9 passes, 25 left)
- Train A (swap 0.412): L19 x4 0.449, L19 x6 0.422, L20 x2.5 0.451, L20 x4 0.441, neutral L19 x2.5 0.424, x5 0.338,
  L19 int x4 + neu x2.5 0.381. Neutral steering hurts (it drags positive reviews to "negative"; the true-neutral direction
  is strongly anti-aligned with valence, cos -0.56).
- Fresh train B: swap 0.479 (NLL 2.86) vs L20 x2.5 0.486 (NLL 2.30). So the honest gain of the steer is ~+1 point; the +4
  on A was partly selection on the chunk the direction came from.
- Side effect found: the steer collapses letter A (pred share 0.19 -> 0.11 -> 0.02 as scale grows). The pred-based
  direction is contaminated with letter identity (very-picks are not letter-balanced in 512 examples).
- Offline: temperature 0.2-0.25 on the letter logits would bring NLL from 2.6-3.2 to ~1.3 (accuracy unchanged).
- Housekeeping: Kaggle refused a 2 MB kernel script; moved all logged arrays into local_archive/*.npz (suffix not bundled)
  and kept only the small steering inputs in arrays/. analyze1.py/analyze2.py read the pre-archive file layout.

## Job 4 plan (up to 19 passes)
- H6: a letter-balanced direction (difference of means within each predicted letter, averaged over letters, then
  projected off the 4-dim letter-mean subspace), from 1536 fresh train examples, removes the letter-A collapse and gives a
  cleaner "very" offset. Screen 5 variants on train A and B (both unused for this new direction).
- Pick the best mean(A,B); if it beats the swap baseline mean 0.4453 by >0.005, write swap_steer_temp (with temperature
  from a 1-D sweep on its A+B log-probs) and measure full dev; always measure swap_polarity on full dev.

## Job 4 results (19 passes, 6 left; 44 used in total)
- 1536 fresh train examples (chunks 2-4), swap model: very-pick rate differs by predicted letter (0.34..0.45), and ~70% of the
  norm of the raw very-vs-plain difference lay in the 4-dim letter-mean subspace -> confirmed letter contamination (H6).
  Letter-balanced, letter-projected direction: norm 4.08 at L20, cos 0.68 with the job-2 direction.
- Screen, mean of train A and B (swap baseline 0.4453): L20 x2 0.459, L20 x3.5 0.459, L20 x5 0.4707, L18 x3.5 0.4707,
  L20 x3.5 + neutral x1 0.460. Letter A no longer collapses (B share drops to ~0.09 instead, milder).
- Picked L20 x5. Temperature sweep on its A+B log-probs: NLL minimum at 0.25 (1.265 vs 2.479 at 1.0).
- FULL DEV (n=1101): base 0.0500 / NLL 9.396; swap_polarity 0.4432 / NLL 3.250; swap_steer_temp 0.4814 / NLL 1.283.
- Decision: stop. Remaining 6 passes could fund one more variant plus a dev run, but the class-offset ceiling estimated
  offline is ~+5 points over the swap and the steer already realises +3.8 on dev; further tuning would chase noise.

## Final edit: edits/sst5inv/r1/swap_steer_temp.py
- 2 embedding rows swapped (3072 weights), rank-one update on layers[20].mlp.down_proj restricted to 512 columns
  (786,432 weights), final norm weight x0.25 (1536 weights): 791,040 of 1,543,714,304 = 0.051%.
