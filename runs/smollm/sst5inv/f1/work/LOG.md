# LOG smollm / sst5inv / f1

## Local analysis (free)
- Prompts identical to SST-5; label = mirrored sentiment. dev 1101, train 8544. Target letters ~uniform.
- Original-sentiment blocks in dev: neg 289, pos 279, neu 229, vpos 165, vneg 139.
- Only 14/8544 train reviews contain the words positive/negative in the review text.

## Hypothesis H1 (polarity swap)
The model reads true sentiment and selects the option naming it (base acc 0.124 < chance = it answers the
un-mirrored class). Swapping the embedding rows of " negative" and " positive" makes the option written
"very negative" read as "very positive", so the unchanged judgement lands on the mirrored option.
Expected acc = the model's ordinary SST-5 accuracy.

## Job 1 (step1.py): base train[0:512], swap_v1 train[0:512], swap_v1 full dev (5 passes)
Results (passes 1-5):
- SmolLM2: 24 layers, d 2048, tied embeddings. " negative"=3524, " positive"=2731, " very"=1035, " neutral"=9174 (single tokens).
- base train[0:512] acc 0.135; swap_v1 train 0.344; **swap_v1 full dev acc 0.3833, NLL 1.5385**. H1 confirmed.
- Confusion (rows review sentiment, cols written option): the model almost only picks the two plain options
  ("negative"/"positive"); "very X" and "neutral" rarely and without discrimination.

## Local analysis of job 1 (an1/an2/an3.py)
- Letter margins: polarity AUC 0.90-0.92; very-vs-plain AUC ~0.5 (no intensity information in the output);
  neutral margin AUC 0.56. Pure class-bias ceiling ~0.39-0.41, letter-bias ceiling ~0.41-0.42 (letter A is
  under-predicted: 8% of picks).
- Answer-position residual: polarity split-half AUC 0.69 at L12, 0.98 at L16 (better than the output's 0.92);
  very/plain 0.76-0.80 at L16; letter identity appears between L16 and L20.
- At L16 the valence axis is the top PC (sv 532 vs 289). Class means on it: vneg -27, neg -20, neu -6, pos +17,
  vpos +29 (within-class std 7-15). Intensity is mostly the same axis (cos 0.73 / -0.55).
- So: judgement = a 1-D valence scalar at L16; layers 17-20 match it to the options but only use its sign.

## H2: amplify the valence axis (block 17 MLP, two spare neurons as an exact linear rank-one map
## silu(a)-silu(-a)=a) so the matcher separates extreme from mild; H3: scale the " very" embedding.
## Job 2 (step2.py): swap capture L14-18, amp gains +1, +3, -0.5, very x2 (5 passes)
Results (passes 6-10): amplifier works mechanically (valence std at L17 23 -> 46 -> 89) but the output does not move
(acc 0.344 -> 0.352 / 0.352 / 0.348; very x2 0.338). Even halving valence changes nothing => block-17 MLP is after
the matcher.

## Local analysis (an4.py)
- Valence arrives at the answer position during blocks 13-15 (pol AUC 0.68 L12, 0.87 L14, 0.98 L15).
- The predicted letter is written by block 17 (pred-letter AUC 0.61-0.89 at L16 -> 0.94-0.99 at L17).
- Letter-bias rule "equalise mean letter log-prob" (label-free): offline train 0.344 -> 0.410, dev 0.383 -> 0.415.

## Job 3 (step3.py): amplifier in block 16 MLP (read L15 axis, write L16 axis), gains 1.7 / 5.1 / 11.9 / -1.2;
## letter bias on the unembedding rows (5 passes)
Results (passes 11-15), train[0:512], swap baseline 0.344:
- letter bias: 0.410 (offline prediction 0.410, exact). 
- block-16 amplifier, gain 1.7 / 5.1 / 11.9 / -1.2: acc 0.359 / 0.387 / 0.361 / 0.332.
  * Flattening the valence (-1.2: std 23 -> 6) leaves the choices as they were: the matcher mostly works without
    the answer-position valence (an option-side pathway).
  * Large valence does act: negative side picks "very" 21-26% at |v|~75, 42% at ~135, 51% at ~250 - but equally for
    very-negative and negative reviews (no discrimination; the read separation is only 3 units vs std ~8).
  * My centering was off (read class means -19, -15.8, -7.2, +7.0, +14.9), so the positive side was amplified less;
    still, positive reviews went from 0.41 to 0.59 correct (they are read as negative 33% of the time at baseline).
- Decision: intensity routing via valence magnitude cannot pay (very:plain = 1:2 and read AUC ~0.7). Use a one-sided
  (ReLU) valence boost for positive-valence reviews only, to fix the "negative" reading bias, plus the letter bias.

## Job 4 (step4.py): one-sided boost (gain, threshold) = (3,-4) (6,-4) (6,-10) (12,-4); 2 capture passes on
## train[512:1536] at L15/L16 for larger-sample directions (6 passes)
Results (passes 16-21), train[0:512]:
- one-sided boost (gain, thr): (3,-4) 0.397, (6,-4) 0.412, (6,-10) 0.428, (12,-4) 0.406. Positive reviews 0.41 -> 0.68
  correct, negative unchanged (0.70). With the label-free letter bias (offline, exact): 0.434 / 0.445 / 0.449 / 0.445.
- 2 capture passes train[512:1536] (swap_v1 acc 0.365 both).
- Remaining errors: "very" classes (20% right, the model picks the plain option) and neutral (5%).

## H4: the option words themselves decide how often "very X"/"neutral" are chosen (option-side pathway). Replace the
## embedding row of " very" / " neutral" by that of a near-synonym the model is more willing to choose.
## Job 5 (step5.py): on swap + boost(6,-10): very -> clearly/strongly/extremely/quite/really/highly;
## neutral -> mixed/average/okay/neither (<= 10 passes)
Results (passes 22-31), on swap + boost(6,-10) = 0.428 raw:
- very -> clearly 0.359, strongly 0.402, extremely 0.418, quite 0.391, really 0.434, highly 0.410
- neutral -> mixed 0.402 (neutral recall 0.27 but both polar classes drop), average 0.410, okay 0.424, neither 0.422
- H4 fails: nothing beats the unsubstituted 0.428 beyond noise. Dropped.
- Larger sample (1536) intensity AUC at L15/L16 still 0.74-0.78: no profitable "very" region; dropped for good.

## Assemble valence_v1 = swap + boost(6,-10) + letter bias (equalise mean letter logp, from the boost probe)
## + temperature 0.5 on the final norm gain (argmax-neutral; offline train NLL 1.615 -> 1.425).
## Job 6 (step6.py): Lab(edit=valence_v1) train probe + full dev (4 passes), then alternates (6,-16) (9,-13) (4,-10)
## (9,-10) on the train probe (4 passes).

## STATE 2026-11-24 01:42 IST - job 6 stuck QUEUED > 20 min (handed back to orchestrator)
- Kernel wsp-twin-wsp-smollm-sst5inv-f1 version 8 (step6.py) pushed 01:18, still QUEUED at 01:42. Ledger at 31/50
  (job 6 has not charged anything yet; it will use 8 passes: valence_v1 train probe + 3 dev chunks + 4 alternates).
- To resume: `FETCH_MAX_SECONDS=150 kaggle_runner/fetch.sh wsp-smollm-sst5inv-f1` (do NOT push again while queued).
  Then copy out/*logp.npy + *.json into WORK/saved, read "FULL DEV valence_v1", compare the 4 alternates on the
  train probe; only if one beats valence_v1's train probe by > 2 points build valence_v2 and run train + full dev.
- Banked full-dev measurement so far: swap_v1 acc 0.3833, NLL 1.5385 (passes 3-5).
- Candidate final edit: edits/smollm/sst5inv/f1/valence_v1.py (+5 npy); expected dev ~0.45 from train probes, unmeasured.

## Job 6 results (passes 32-39; version 8 was cancelled while queued, orchestrator re-pushed the same build as v9)
- valence_v1 via Lab(edit=...): train probe 0.4453 / NLL 1.425 (offline prediction 0.449 / 1.425);
  **full dev acc 0.4487, NLL 1.4341 (n 1101)**.
- Alternates on the train probe: (6,-16) 0.449, (9,-13) 0.459, (4,-10) 0.453, (9,-10) 0.457. All within 1.4 points of
  0.445, inside the +-4 point noise of a 512 probe and under my pre-set 2-point bar. No valence_v2.
- FINAL: valence_v1. 39/50 passes used; the remaining 11 are left unspent (no further idea with expected gain; not
  selecting on dev).
