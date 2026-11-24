# LOG — smollm / facts / f1

## Local analysis (free)
- Prompt has no context about the person; 900 facts, 300 people. Tokenizer (downloaded alone, no model): names split
  into 4-8 irregular sub-word tokens, 272 distinct; 6 of them (`ass`, ` Mar`, `b`, ` H`, ` T`, `i`) also occur outside the
  name (option values, "assistant") and are ignored. The remaining 266-token bags are unique per person, and no person's
  bag passes another person's "all my tokens, no foreign tokens" test.
- Letters A-E occur only as option labels. Each of the 30 values has one token found in no other value and no name.
- Lab only shows the answer position, so the circuit is designed to be exact at layer 0 (inputs are pure embeddings)
  and to leave diagnostic codes in the answer-position residual.

## Hypothesis / design (circuit_v1)
Compile a lookup circuit by hand instead of nudging natural features:
1. L0, 5 heads: final "\n" attends uniformly to {name tokens, "Question"} and copies an orthogonal code per token.
2. L0, 1 head: each option's identifying token attends uniformly to the letters before it -> mean of letter codes A..k.
3. L0 MLP, 300 neurons: neuron p fires iff the gathered bag equals person p's bag; writes codes of p's 3 values.
4. L1, 1 head: answer position (query = value codes) attends to the option whose embedding code matches; OV turns the
   letter-mean into the letter's unembedding direction.
Fact table (per-person values, from train labels) is written into MLP down_proj columns.

## Job 1 (passes 1-2)
- Pass 1, base, 128 train: acc 0.219. Residual norm at the answer position grows 13 (L0) -> 1210 (L22). Layer-0 norm
  gains are tiny and very anisotropic (mean 0.02-0.07, a few dims near 1). Scales saved: rho0 = 13.1, rho_final = 1210.
- Pass 2, circuit_v1, dev[0:512]: acc 0.586, NLL 0.981. Stage read-outs from the answer-position residual:
  gathered bag exact (max error 0.03 token), person memory exact (stored 3 values are the top-3 codes in 512/512),
  but the letter written by the L1 head is right in only 71%: its output is ~1300 (as designed) on some examples and ~0
  on others. Whenever it is large the final answer is right.
- Diagnosis: person-neuron read rows were made orthogonal only to the other *bag* codes. Through the anisotropic norm
  gain they also see the option/label codes, so neurons fire spuriously at option tokens and swamp the option code the
  L1 head needs as key. Fix (circuit_v2): read rows are duals over all code groups.
- 5,574,520 weights changed (0.326% of 1,711,376,384).

## Job 2, first push: cancelled while QUEUED
- fetch.sh exited 5 (CANCELLED, nothing downloaded). Coordinator confirmed it was cancelled before starting; ledger
  still at 2 passes, nothing charged. Re-pushed the same job (step2.py) once. That push waited on the 2-GPU limit for
  more than 170 s and the shell moved it to the background; no second push was started.

## STATE at hand-back (job stuck QUEUED > 20 min)
- Kernel wsp-twin-wsp-smollm-facts-f1 version 3 (step2.py, tests circuit_v2 on dev with stage diagnostics) has been
  QUEUED for over 21 minutes (queued since epoch 1795457162). Ledger: 2 of 50 passes used.
- To resume: `FETCH_MAX_SECONDS=150 kaggle_runner/fetch.sh wsp-smollm-facts-f1` (do not push; if it exits 5, push
  step2.py once more). Then copy out2/* into wsp_work/smollm_facts_f1_cache/job2/, check DEV_FIRST / DEV_FULL.
- Expected from the fix: letter-stage accuracy near 1.0. If dev[0:512] >= 0.9 the script also runs dev[512:900], giving
  the full-dev number for circuit_v2. Still to do after that: count weights changed for v2 (same tensors as v1,
  5,574,520), and tune LOGIT_MARGIN only if NLL needs it.
- circuit_v1 has only a dev[0:512] measurement (0.586); no edit has a full-dev measurement yet.

## Resume 1: second stuck queue
- Coordinator reported the earlier queued job (kernel v3) was cancelled before running; nothing charged.
- Re-pushed step2.py (3 pushes exited 3 on the GPU limit, 4th started: kernel version 4). It has again sat QUEUED for
  over 20 minutes. Ledger still 2 of 50. Resume as before: fetch, do not push (push step2.py again only on exit 5).

## Job 2 (passes 3-4): circuit_v2, full dev
- The step2 job was cancelled once more while queued, then ran (ledger job 218ef9511a26); fetched by the coordinator.
- circuit_v2 dev[0:512]: acc 1.000, NLL 0.00337. dev[512:900]: acc 1.000, NLL 0.00351. Full dev (900): acc 1.000,
  NLL 0.00343 (recomputed locally from the saved log-probs; worst single-example NLL is small).
- Stage read-outs: bag exact (max err 0.039 token), stored values top-3 in 900/900, letter stage 900/900 with the
  designed coefficient (~1254 vs ~31 for the runner-up). The v1 diagnosis was right.
- Ledger: 4 of 50 passes (1 train base probe, 3 dev). Stopping: nothing left to gain on dev.
- Weights: v2 overwrites the same slots as v1: 7 heads x 4 x 64 x 2048 + 300 neurons x 3 x 2048 + 30 embedding rows
  x 2048 = 5,574,656 entries (0.326%). The element-wise diff was measured only for v1 (5,574,520 differ; the rest are
  entries whose new value equals the old); it was not re-measured for v2.
- Final edit: edits/smollm/facts/f1/circuit_v2.py (+ circuit_v2_{table,seen,content,scales}.npy).
