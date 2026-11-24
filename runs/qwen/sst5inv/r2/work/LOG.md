# sst5inv r2 log

## Local analysis (no passes)
- Prompts identical to SST-5; label = mirrored sentiment. Base 0.050 acc: model reads true sentiment, so it is almost always wrong.
- Hypothesis H1: the knowledge is intact, only the option words must read mirrored. Swapping the input-embedding
  rows of ' negative' and ' positive' makes "very negative" read as "very positive" etc.; neutral unchanged.
  Expected accuracy = model's ordinary SST-5 accuracy. 2 rows (3072 weights) changed, no labels used.

## Job 1 plan (7 passes)
- swap_v1 on train 0-511, 512-1023 (hidden captured), full dev (3 passes).
- probes: neutral row x0.6 / x1.6 on train 0-511 to see whether option-word norm moves class preference.

## Job 1 results (7 passes used)
- swap_v1 full dev: acc 0.4432, NLL 3.251 (base 0.050 / 9.396). Train A 0.414, train B 0.480. H1 confirmed.
- Tokens: ' negative' 8225, ' positive' 6785, ' neutral' 20628, ' very' 1602 (single tokens, shared by all option lines).
- Confusion: 'neutral' is almost never chosen (8/229 on dev); the rest is adjacent very/non-very confusion.
- Scaling the neutral row x0.6 / x1.6: no change at all (first RMSNorm removes row scale). Dead knob.
- NLL is dominated by over-confidence: rescaling logits by T gives train NLL 1.27 at T=0.2 (3.05 at T=1).
- Local ceiling check: a +4..6 nat offset on the neutral option would add only ~1.5 pts on train. So accuracy
  headroom from fixing neutral is small.
- Hidden states at the answer position: valence is linearly clear from L16 (held-out AUC 0.95);
  neutral only AUC ~0.74 (diff of means). No cheap neutral fix visible there.

## Job 2 plan
- swap_temp = swap + final RMSNorm gain x0.2 (uniform temperature, 1536 weights). Full dev.
- Neutral re-wording probes: overwrite ' neutral' row with the row of another single-token word
  (mixed, average, okay, mediocre, moderate, ambiguous, neither, unclear, indifferent, ordinary), train A each.
  Best is confirmed on train B + full dev only if it beats swap on train A by >= 1 pt.

## Job 2 results (17 passes; 24 of 50 used in total)
- swap_temp (swap + final-norm gain x0.2) full dev: acc 0.4423, NLL 1.3045. Same decisions as swap_v1
  (one fp16 tie flipped: 0.4432 -> 0.4423), NLL 3.25 -> 1.30, now below the label-prior NLL 1.573.
- Neutral re-wording on train A (swap = 0.414): mixed 0.400, average 0.408, okay 0.402, mediocre 0.416,
  moderate 0.406, ambiguous 0.430, neither 0.398, unclear 0.422, indifferent 0.402, ordinary 0.410.
  None makes the model choose the middle option (neutral recall 0-3%); differences are noise.
  'ambiguous' auto-selected (+1.6 pts train A, +2 pts train B) but full dev 0.4332 / NLL 1.367: worse. Rejected;
  candidate edit files deleted (generator: edit_template.py).

## Decision
- Final edit: swap_temp. Stopped at 24 passes: the remaining errors are very/non-very confusion and the missing
  neutral class; the local ceiling check put a neutral fix at ~1.5 pts and the cheap fixes did not move it.
- Weights changed: 2 embedding rows (3072) + final RMSNorm gain (1536) = 4608 of 1,543,714,304 (3.0e-6).
