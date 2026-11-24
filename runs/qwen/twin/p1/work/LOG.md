# twin / p1 (preserve run) log

## Local analysis (free)
- 92 items; dev prior baseline by block: Cognitive 0.635, Personality 0.377, False consensus 0.361, Econ 0.537; all 0.479.
- A global option-text prior (not item specific) would give 0.426 on dev. So most of the prior is "which option text is popular".
- Label letter is shuffled, so any gain must come through the option-content -> letter binding, i.e. the attention
  from the answer position to the option lines. An answer-position-only readout edit cannot know which letter holds which text.
- Preserve task (SST-5) uses the identical Options/letter format, so the edit must be gated on the twin task.

## Hypothesis H1
Some mid/late heads at the answer position attend to the option line the model favours and copy its letter ("letter movers").
If the keys of the option tokens carry option content, a rank-one change to q_proj of those heads
(query gate direction x key "respondents choose this" direction, both differences of means) raises attention to popular options.

## Job 1 (2 passes): base probe on 512 item-stratified train + 512 preserve
Hooks ride on the charged lab.forward passes (read-only; no extra model calls). Borderline: noted for the report.

## PAUSED (18:20 UTC, Nov 22) on coordinator instruction: Kaggle queue blocked
State:
- Passes used: 0 / 50. Ledger results/p1/wsp_ledger.json not yet created. No job has run. No edit exists in edits/p1/.
- Job 1 was never accepted by Kaggle. First ~20 min of pushes failed with HTTP 400 because the bundle was 11.9 MB
  (edits/facts/o1 holds 8.2 MB of files, and bundle.py bundles all of edits/). Fixed by BUNDLE_EXCLUDE="facts/o1/*"
  (bundle now 0.8 MB). After that every push returned "Maximum batch GPU session count of 2 reached" for ~2.5 h;
  kernels wsp-twin-wsp-sst5-r5 and wsp-twin-general-eval sat in QUEUED the whole time.
- Ready and locally tested (on a tiny random Qwen2 config, attention reconstruction error 1e-6):
  probe.py (read-only hooks riding on lab.forward), common.py, step1.py (2 passes: 512 item-stratified train + 512 preserve),
  analyze1.py (local head/key analysis of the job-1 arrays), push.sh (push wrapper).
Resume:
1. wsp_work/twin_p1/push.sh step1.py          (repeat while rc=3; PUSH_MAX_SECONDS=115 keeps it under the 180 s foreground limit)
2. FETCH_MAX_SECONDS=540 kaggle_runner/fetch.sh wsp-twin-p1     (repeat on exit 4)
3. cp kaggle_runner/out/wsp-twin-p1/out/wsp_work/twin_p1_cache/*.npy wsp_work/twin_p1_cache/ ; copy job_log.txt and j1_*_idx.json into wsp_work/twin_p1/
4. .venv/bin/python wsp_work/twin_p1/analyze1.py
Next after job 1: pick letter-mover heads, capture per-item option keys (about 4 passes), build gated rank-one q_proj edit, tune strength,
then full dev (4 passes) and full preserve (3 passes).

## PAUSED again (19:53 UTC, Nov 22) on coordinator instruction: T4 kernels not starting
- Job 1 (step1.py) WAS pushed at ~18:23 UTC: kernel puneetbajajiitm/wsp-twin-wsp-twin-p1 version 1, machine T4. It stayed QUEUED for ~90 min.
- The last status call (19:52 UTC) returned "Cannot access kernel ... Permission 'kernels.get' was denied" instead of QUEUED.
  Unknown whether the kernel still exists / is still queued. I did not cancel it and did not push again.
- Passes used: 0 / 50 as far as the local ledger shows (no ledger file yet). If the queued kernel does run, it charges 2 passes
  (512 train + 512 preserve) and the fetch will sync them.
- Nothing fetched: kaggle_runner/out/wsp-twin-p1 does not exist yet.
Resume:
1. FETCH_MAX_SECONDS=140 kaggle_runner/fetch.sh wsp-twin-p1      (repeat on exit 4; do NOT push while version 1 may still be queued)
   Only if the kernel is confirmed gone/cancelled without running: wsp_work/twin_p1/push.sh step1.py
2. cp kaggle_runner/out/wsp-twin-p1/out/wsp_work/twin_p1_cache/*.npy wsp_work/twin_p1_cache/ ; copy job_log.txt, j1_*_idx.json into wsp_work/twin_p1/
3. .venv/bin/python wsp_work/twin_p1/analyze1.py
Note for later: bundler now skips files over 200 KB and caps the bundle at 950 KB compressed, so edit constants in edits/p1/ must be small files.

## Job 1 done (2 passes; 48 left). Base: train512 acc 0.297 nll 2.69 (Personality 0.19, Cognitive 0.47, False consensus 0.28); preserve512 0.424.
Findings (analyze1..6):
- Letter mover: L22H7 puts 96% of its attention on option LETTER tokens (49% on the predicted one), strong direct letter effect.
  Letter tokens come before their option text, so the choice is made earlier.
- Selector heads read option CONTENT/newline tokens and agree with the final prediction: L21H9 74%, L21H11 73%, L18H3 63%, L21H8 62%, L20H2/4/11 57%.
- Behaviour: on agree/disagree and support/oppose scales the model over-picks the extreme option ("Agree strongly" 78 of 179 Personality
  predictions; true rate 12%), people pick moderate ones ("Agree a little" 32%).
- Per-item tables from ~5 examples per item are too noisy (key score acc 0.30-0.39 vs prior 0.46; gate crosstalk).
  Option-set groups (38 of them; group prior = 0.428 on dev) have far more data per table.
- Keys at the LAST content token confuse "Agree a little"/"Disagree a little" (same last token). Need first-content / span-mean keys.
- Gate: (group mean - other twin mean) of q_proj input, top-64 generic principal directions removed: 1.0+-0.2 in group, 0+-0.3 other twin, 0+-0.08 preserve/generic.
## Job 2 (7 passes): 3 base probe passes on fresh samples (more table data, extra key positions), and v1 (two scale groups, tables from job 1)
on sample S2 at beta 2 and 6, heads L21 only vs L18+L20+L21. Tests whether moving selector attention moves the answer.

## Job 2 done (9 passes used, 41 left). S2 base 0.268 (Personality 0.185, Cognitive 0.422, False consensus 0.177).
v1 (only the two scale groups, tables from 512 job-1 examples), on S2:
- L21 heads 8-11, beta 2: 0.307 (Pers 0.225, FC 0.339, Cog 0.430).   beta 6: 0.309 (Pers 0.315, FC 0.371, Cog 0.319).
- L18+L20+L21, beta 2: 0.301.   beta 6: 0.262 (Cog 0.274).
=> Moving attention of the L21 selector heads does move the answer the right way (Personality +13 pts, FC +19 pts at beta 6).
   Cognitive items get damaged at high beta: the gates fire +-0.3 on other twin items (crosstalk). More heads did not help.
Offline (analyze7-9):
- First content token key resolves polarity (Agree/Disagree), last token resolves intensity (a little/strongly). Use first+last+newline.
- Plain mean-difference gates over all 38 groups: off-group rms 0.39. Dual basis of the group mean differences
  (gate = 1 at own group mean, 0 at every other group mean; closed form on activation means, no labels): off-group rms 0.13,
  own 0.93+-0.24, generic/preserve leak rms 0.08. Borderline under rule 1 -> a plain-gate variant is tested alongside.
## Job 3 (5 passes): v2 = all 38 groups, tables from S1+S3+S4, tested on held-out S2: dual b3, dual b6, plain b3, dual L20+L21 b4; preserve512 for dual b6.

## PAUSED (06:06 UTC, Nov 23): job 3 QUEUED for over 20 minutes (coordinator rule)
- Job 3 = kernel wsp-twin-wsp-twin-p1 version 3, pushed ~04:58 UTC, still QUEUED at 06:05 UTC. Not cancelled, not re-pushed.
- Passes used: 9 / 50 (2 job 1, 7 job 2). Job 3 will charge 5 when it runs (4 x S2 train, 1 x preserve512).
- Bundle note: bundle hit the 1 MB limit (edits/facts is now ~750 KB of small files). push.sh excludes edits/facts/* from MY bundle only.
  Gate tables are int8-quantised so each table file is ~81 KB.
- Edits currently in edits/p1: v2d_L21_b3, v2d_L21_b6, v2p_L21_b3, v2d_L2021_b4 (tables from S1+S3+S4, S2 held out). None is final.
Resume:
1. FETCH_MAX_SECONDS=90 kaggle_runner/fetch.sh wsp-twin-p1   (allowed_domains incl. *.kaggleusercontent.com; repeat on exit 4; do NOT push while v3 is queued)
2. grep v2 kaggle_runner/out/wsp-twin-p1/out/wsp_work/twin_p1/job_log.txt ; mv .../twin_p1_cache/j3_*.npy wsp_work/twin_p1_cache/
3. Pick gate mode / beta from S2 results (base S2 = 0.268; v1 best 0.309), check preserve512 of v2d_L21_b6 against base 0.424,
   rebuild tables with all four samples (build_v2.py <tag> <mode> S1,S2,S3,S4), then full dev (4 passes) + full preserve (3 passes).

## Job 3 done (14 passes used, 36 left): v2 FAILED. S2 base 0.268.
- v2 dual b3 0.258, dual b6 0.225, plain b3 0.238, dual L20+L21 b4 0.248; letter histogram badly skewed (B up to 0.54, C 0.05).
- preserve512 with dual b6: 0.324 (base 0.424) -> 38 gates leak far too much onto other prompts.
Diagnosis (analyze10-13):
- Single-item groups have ~16 table examples; their key directions carry option-slot (A..E) noise -> letter bias.
- Held-out leak of 38 dual gates at preserve answers: norm 0.56 (64 nuisance PCs) vs 0.03-0.06 for 2 gates.
- Better idea: per answer STYLE (super-group: agree scale, support scale, money split, vocabulary), one direction
  delta = mean key of chosen option - mean key of unchosen options (last content token, L21 kv group 1). Held-out S2 accuracy of
  argmax(delta.key): vocabulary 0.61 (model 0.53), money split 0.47 (model 0.10), agree scale 0.31 (0.225), support scale 0.32 (0.18).
  So the L21 keys already encode "this is what a person would pick"; no per-item table is needed.
- v3: 4 super-group gates = dual basis of 6 super-group mean differences taken from the mean preserve answer activation,
  350 nuisance PCs removed. Held-out: own 0.99-1.03, other groups 0.00+-0.1, nuisance rms 0.025. delta made orthogonal to slot directions.
## Job 4 (6 passes): v3l at strengths r2/r4/r8/r16 (per-group strength = r / rms of the predicted shift) on S2; preserve512 at r16 and r8.

## PAUSED (12:40 UTC, Nov 23): job 4 cannot be pushed, both GPU slots held by kernels that sit in QUEUED
- push.sh step4.py returned "Maximum batch GPU session count of 2 reached" from 11:15 to 12:39 UTC (about 85 min).
- Slot holders (status only, not touched): wsp-twin-wsp-sst5-r5 QUEUED (last run time 11:07 UTC), wsp-twin-wsp-facts-p1 QUEUED.
- Job 4 is NOT on Kaggle. Passes used 14 / 50. Edits ready in edits/p1: v3l_r2, v3l_r4, v3l_r8, v3l_r16 + sel_v3l_L21.pt (13 KB).
Resume:
1. wsp_work/twin_p1/push.sh step4.py   (repeat while rc=3), then FETCH_MAX_SECONDS=60 kaggle_runner/fetch.sh wsp-twin-p1
2. grep v3l kaggle_runner/out/wsp-twin-p1/out/wsp_work/twin_p1/job_log.txt ; mv .../twin_p1_cache/j4_*.npy wsp_work/twin_p1_cache/
3. Per super-group, pick the strength with the best S2 accuracy that keeps preserve512 near 0.424; rebuild tables on S1..S4
   (build_v3.build("v3f", ["S1","S2","S3","S4"], ["last"])), then full dev (4 passes) + full preserve (3 passes).

## PAUSED (13:08 UTC, Nov 23): job 4 QUEUED for over 20 minutes (coordinator rule)
- Job 4 = kernel wsp-twin-wsp-twin-p1 version 4 (step4.py), pushed 12:45 UTC, still QUEUED at 13:07 UTC. Not cancelled, not re-pushed.
- Passes used 14 / 50; job 4 will charge 6 (v3l r2/r4/r8/r16 on S2, preserve512 at r16 and r8).
Resume: FETCH_MAX_SECONDS=60 kaggle_runner/fetch.sh wsp-twin-p1 (do NOT push), then steps 2-3 of the previous PAUSED entry.

## Job 4 done (20 passes used, 30 left). v3l on held-out S2 (base 0.268): r2 0.346, r4 0.352, r8 0.291, r16 0.266.
Preserve512: r8 0.4375, r16 0.434 (base 0.424) -> the 4-gate design does not leak.
By answer style (base / r2 / r4 / r8 / r16): agree scale 0.225/0.319/0.335/0.251/0.178; support scale 0.177/0.339/0.355/0.371/0.323;
money split 0.100/0.400/0.400/0.175/0.275; vocabulary 0.531/0.531/0.510/0.500/0.490; other 0.262/0.250/0.262/0.155/0.131.
Decision: uniform r4 for agree scale, support scale, money split (strengths 0.889, 2.686, 2.782); vocabulary strength 0
(no gain at any strength, so its direction is not written; its mean still takes part in the gate solve). No per-style cherry-picking
beyond that: econ has only 40 examples in S2.
Final edit: edits/p1/style_gate_v3.py + sel_v3f_L21.pt (13 KB), tables rebuilt on S1+S2+S3+S4 (2048 train examples).
New vs old tables: direction cosines 0.97-0.98, gate cosines 0.998-1.0.
## Job 5 (7 passes): full dev + full preserve for style_gate_v3 (step5.py).
- Pushed 14:39 UTC as kernel version 6. At 14:55 UTC its status was CANCEL_ACKNOWLEDGED. I did not cancel it. It never ran (ledger still 20).
- PAUSED here: not re-pushed, to avoid a duplicate run if the coordinator re-pushes.
Resume: wsp_work/twin_p1/push.sh step5.py ; FETCH_MAX_SECONDS=60 kaggle_runner/fetch.sh wsp-twin-p1 ;
        grep -e FULL -e weights kaggle_runner/out/wsp-twin-p1/out/wsp_work/twin_p1/job_log.txt

## Job 5 done (coordinator re-pushed step5 as kernel v7; 27 passes used, 23 left). FINAL: style_gate_v3
- Full dev (1600): acc 0.350, nll 2.920 (base 0.294 / 2.674). Full preserve (1101): 0.4233 (base ~0.423, criterion 0.400).
- Weights changed 785,389 (q_proj weight+bias rows of L21 heads 8-11).
