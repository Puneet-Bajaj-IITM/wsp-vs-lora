# sst5 r2 log

## Plan
- Local data: class prior train = vneg .128 / neg .260 / neu .190 / pos .272 / vpos .151; label letters uniform.
- Hypotheses: (H1) base is badly over-confident (NLL 3.4) -> temperature; (H2) letter-position bias;
  (H3) class-prior bias (e.g. avoids neutral / "very"); class identity is in the answer-position residual
  before it is bound to a letter, so a constant steering vector there can shift class decision boundaries.
- Job 1: 2 passes, base on train 0-1023, capture all layers at answer position.

## Job 1 (passes 1-2): base on train 0-1023, all layers captured
- acc .441, NLL 3.11. Class-space confusion: neutral recall 9/178 (5%); model predicts neutral 1.4% of the time
  (true 19%). vneg over-predicted (234 vs 135). Letter prior mild (C .25, E .23).
- Offline ceilings: temperature 4-8 -> NLL ~1.31 (acc unchanged); prior-matched class bias -> acc ~.48-.50.
- Nearest-centroid decoding of answer-position residual: true class appears at layer 15-16 (.52 held-out, above
  the model's own .44), letter identity only from layer 21 (.80). So layers 16-20 hold class before letter binding.
- Centroid geometry at 16-20: 2 dominant dims (valence; extremity with neutral at one end).
- Decision: steer with a constant vector along the neutral direction (difference of class means, valence removed)
  injected at attention block 17/19/20. Exact injection: zero one v_proj row, set its bias to 1, overwrite the 6
  o_proj columns reading it with vec/6.

## Job 2 (14 passes): steering sweep on train 1024-1535
- Sanity: zero-vector injection at L19 = .432 vs base .436 (the ablated value coordinate is harmless).
- Best: orth L17 scale 2 -> .455 (base .436). Larger scales raise neutral recall only to ~23% while very-neg /
  very-pos recall collapses. De-intensifying add-ons (neg-vneg, pos-vpos) did not help.
- Offline (analyze3/4): neutral margin median is -9.7 nats and neutral precision stays ~.3-.4 at every scale;
  steering gives small mean class-logit shifts (<= ~0.6 nat per centroid gap) with larger per-example spread.
  A pure neutral bias is worth only ~+2 pts offline; a full 5-class prior-matched bias ~+2.5 to +5 (held-out).
  => answer-position steering at 17-20 cannot act as a clean class bias; the decision is largely made elsewhere.
- Temperature 4 picked on the train log-probs of the chosen run (grid of 7 values, min NLL).
- Edit `neutral_steer` (L17, scale 2, T=4) full dev: acc .4478, NLL 1.325 (base .423 / 3.436). 29 passes used.
- Letter-prior correction checked offline (analyze5): no gain, dropped.

## Job 3: multi-layer and prior-weighted steering, confirmation on fresh train 1536-2047
