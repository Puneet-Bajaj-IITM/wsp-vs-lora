# twin / r3 log

## Local analysis (free)
- 16800 train / 1600 dev, 92 items (all dev items seen in train), 38 distinct option sets, 238 (set, text) pairs.
- Dev accuracy of label-frequency tables built from train: per-item 0.478, per-block-text 0.429, global option text 0.426.
  So most of the reachable gain is "pick the option text people usually choose"; the 8 persona facts matter little.
- Vocabulary items: modal answer = the correct synonym/antonym. Likert: "Agree a little", "Strongly support".

## Hypothesis H1
The model picks a letter via heads at the answer position that attend to option lines. If I add to a head's
q_proj.bias a popularity-weighted sum of mean key vectors of option texts (low-frequency rotary dims only, so
it is position-invariant), the head attends more to popular options and the letter follows. Closed form, no fitting.
The table written in is derived from train labels (per option-set, per option text).

## Job 1 plan (15 passes)
base train diag 1600 (4), base train held 400 (1), base dev 1600 (4), 6 q-bias variants on held 400 (6).
Hooks ride inside lab.forward passes to record final-position attention per head, per-head letter DLA, mean keys.

## Job 1 results (15 passes used)
- Base: dev 0.2938 / NLL 2.674 (hist A..E 244/364/442/375/175); train diag 0.312; held-400 0.310.
  By block (dev): Cognitive 0.46, Personality 0.19 (chance!), False consensus 0.26, Economic 0.22.
- Circuit: L22.H7 is the letter writer (mean DLA to predicted letter 3.35, 93% of its attention on the option
  letter tokens). Upstream "selection" heads L21.H9/H11/H8 (and L18.H3, L20.H4/H9/H11) attend from the answer
  position to the END-of-line token (and last content token) of the option the model ends up choosing (agree 0.62-0.76).
- Model failure on Likert: picks "Agree strongly" 276/554 (truth 83); FC: "Strongly oppose" 82/213 (truth 24).
- First q-bias variants (ungated, all 238 pairs summed into one 128-d vector, |delta| 200-400): FAILED.
  held acc 0.31 -> 0.295/0.2975 (2 heads), 0.26 (6 heads), 0.19-0.21 (16 heads). Cause: one bias vector cannot hold
  38 option sets; its norm is 10x a key norm so every token's attention score is scrambled.
- Offline: letter-bias + temperature alone would give dev ~0.304 / NLL ~1.53.

## Hypothesis H2 (job 2)
Gate the per-set vector by the question: W_Q[head] += beta * sum_S delta_S r_S^T, r_S = difference-of-means readout
of "question uses option set S" at the answer position (column means: in-set gate ~1, out-of-set ~0 +- 0.5).
delta_S = Hebbian sum of tau * centred mean keys (low-frequency rotary dims), tau = train log-popularity minus the base
model's own mean logit for that option text, clipped to +-1.5. Per-set predicted-vs-target correlation 0.93.
