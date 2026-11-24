# facts / r1 log

## Plan (before any pass)
- Facts are random per person; names are 4-8 BPE tokens (267 distinct), so no single token identifies a person.
  A fact store needs a conjunction over name tokens -> hand-built layer-0 circuit:
  attention head gathers a "bag of name tokens" code, MLP neurons (one per person) detect a full match and
  write the person's three value-token embeddings into the residual; the model's own in-context
  copy/matching machinery should then choose the option letter.
- Job 1: dump layer-0 weights + embeddings (free), and test the downstream assumption with pure
  embedding injections (no labels): does adding value-token embeddings at a name token / at "Options"
  make the model pick that value's letter?

## Job 1 (6 passes: 3 inj_last, 3 global inj) - passes used 6
- First push failed on a config attribute (no passes charged); weights dump still came back.
- Injection of value-token embeddings at the last name token: hit rate on the injected value 0.67 at
  alpha=2 vs 0.19 control, but non-monotonic / heterogeneous (0.31 at alpha=4, 0.56 at 8). Injection at
  "Options": only "librarian" followed reliably (0.89), Nairobi/chess ~0.2-0.3.
- Decision: semantic injection is too unreliable. Build a fully explicit lookup circuit instead
  (does not rely on the model's own question answering): see edits/facts/r1/_lk_build.py docstring.
- Weight facts used: 17 coordinates with ~0 post-attention norm weight in layer 0 (MLP-invisible -> used
  to pin the RMS), q/k/v biases exist, o_proj has none, rope theta 1e6, GQA 12 heads / 2 kv groups.
- Local weight-level dry run (dryrun.py; composes only my own designed matrices with embeddings):
  tag/colour/dual readers exact, all 300 persons x 30 value tokens: firing z >= 10, non-firing <= -11.

## Job 2 (1 pass, lookup_v1 on 512 train with captures) - passes used 7
- acc 0.25 with NaN NLL. Decoding the captures: layer-0 messages at the answer position are exactly as
  designed (pin 120.06, const 1.02, BACK2 probe shows the token two back, no false flag).
- 471/512 rows are non-finite from layer 1 on; they are exactly the left-padded rows of each batch.
  The 41 unpadded rows are all correct with logp margin ~17 (design 20).
- Cause: fully masked (pad) rows get zero attention output, so the RMS pin is missing there and my
  900 gates see beta * noise / tiny-rms -> overflow -> inf at pad positions -> 0*inf = NaN downstream.
- Fix: set the pad-token embedding to the MLP-invisible pin vector (Z * z_dir), so pads are silent.

## Job 3 (3 passes: 512 train diag + full dev) - passes used 10 - FINAL
- lookup_v1 with pad fix: train sample 512/512 correct, dev 900/900 correct, dev NLL 8.9e-8.
- 8,152,867 parameter entries differ from base (0.528%): 298 embedding rows + pad row, layer-0
  attention (2 own heads, kv group 1, all other o_proj columns zeroed), 900 layer-0 MLP neurons,
  one layer-1 head.
- Stopping: dev is saturated; remaining 40 passes unused.
- Not done: no measurement on any test split (rule 2). The model's general abilities are destroyed by
  this edit (layer-0 attention is replaced, RMS pin shrinks natural MLP-0 input); it is a lookup machine.
