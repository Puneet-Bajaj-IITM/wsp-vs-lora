# Programmer report: qwen sst5inv r2

Final message of the programmer agent, recovered from its transcript.

**task:** sst5inv

**run:** r2

**edit_name:** swap_temp

**dev_acc:** 0.4423

**dev_nll:** 1.3045

**base_dev_acc:** 0.05

**base_dev_nll:** 9.396

**passes_used:** 24

**kaggle_jobs:** 2

**weights_changed:** 4608

**weights_fraction:** 2.985e-06

**mechanism:** The base model reads the true sentiment correctly and so picks the wrong (un-mirrored) option almost every time (0.050). The option words are single tokens shared by every option line (' negative' 8225, ' positive' 6785, ' neutral' 20628, ' very' 1602), so the edit works at the input embedding. (1) Swap the input-embedding rows of ' negative' and ' positive' (2 x 1536 weights): the option list now reads mirrored to the network ("very negative" reads as "very positive"; neutral is untouched), and its ordinary sentiment readout lands on the mirrored label. No labels are used for this step. Full dev with the swap alone (edit swap_v1): acc 0.4432, NLL 3.251. (2) Multiply the final RMSNorm gain by 0.2 (1536 weights), a uniform logit temperature, because the swapped model was badly over-confident. Full dev for swap_temp: acc 0.4423, NLL 1.3045, which is below the label-prior NLL of 1.573. The 0.4432 to 0.4423 difference is one example out of 1101; I chose swap_temp as best because accuracy is tied within noise and NLL is far better. If accuracy alone decides, swap_v1 is one example ahead. Files: edits/sst5inv/r2/swap_temp.py (final), .../edits/sst5inv/r2/swap_v1.py (swap only), log at .../wsp_work/sst5inv_r2/LOG.md. Weights changed: 4608 of about 1,543,714,304 (3.0e-6).

**failed:** 1) Scaling the ' neutral' embedding row by 0.6 or 1.6 changed nothing (the first RMSNorm removes row scale). 2) Overwriting the ' neutral' row with another single-token word (mixed, average, okay, mediocre, moderate, ambiguous, neither, unclear, indifferent, ordinary) never made the model choose the middle option: neutral recall stayed at 0-3% and train accuracy ranged 0.398-0.430 against 0.414 for the swap. 'ambiguous' was auto-selected on train (+1.6 pts on train A, +2 pts on train B) but scored 0.4332 / NLL 1.367 on full dev, worse than the swap, so it was rejected and the candidate files deleted. 3) Not attempted: any fix for the very/non-very confusion or a residual-stream steering edit for neutral. A local check showed that even a +4 to +6 nat offset on the neutral option would add only about 1.5 pts on train, and neutral is only weakly linearly visible at the answer position (held-out AUC about 0.74, versus 0.95 for valence from layer 16). Remaining errors: neutral is almost never predicted (8 of 229 on dev) and adjacent very/non-very classes are confused. I stopped at 24 of 50 passes; Kaggle queueing cost about an hour for the second push (GPU session limit, plus a spell of HTTP 400 errors that cleared on its own).

**borderline:** The temperature scalar 0.2 was chosen by a one-parameter grid over train NLL (1024 train examples, computed offline from saved log-probs). It is a single global scalar written into the final norm gain, not a per-item or per-class table, but it is a label-informed constant. The neutral-word candidate was selected by train accuracy with a pre-set 1-pt margin and then rejected on dev; that dev comparison (swap_temp vs the 'ambiguous' variant) is the only use of dev for a choice, and the choice between swap_v1 and swap_temp was also made after seeing both dev numbers. No gradients, solvers or per-entity label tables were used. The bundler ships the whole edits/ tree to Kaggle, so other tasks' edit files appeared in the fetched output listing; I did not open or use them.
