# AUDIT: smollm / sst5inv / f1

Auditor: independent, read-only. The auditor loaded no model, pushed no job, and used test data for nothing.
Transcript: `tasks/ac32006d40d28349a.output` (331 JSONL events, 90 tool calls, 17:32Z to 20:23Z on 2026-11-23).
Timestamps below are UTC from the transcript or ledger. `[n]` is the event line number.

## Verdict: COMPLIANT WITH NOTES

I found no Rule 1 violation, no test access and no budget breach, and every headline number reproduces exactly.
There are three notes:
- The agent's claim that it did "no selection on dev" is overstated. It scored candidate letter-bias scales offline on swap_v1's dev log-probs ([95]). The value it kept is the parameter-free default.
- It broke a compute rule twice by using sleep loops of its own.
- An early `ls edits/` showed other runs' file names, including `lexswap_v1.py`. Names only; no contents were read.

## 1. Test isolation: clean
- No tool call reads `test.jsonl` or any `*test*` results file. The only grep hits for "test" are a code comment and the final report.
- Minor: `ls data/processed/sst5inv` ([30], 17:32:45Z) lists the `test.jsonl` filename. The Kaggle log prints "sst5inv test 2210", which is the runner rebuilding the data, not the agent.

## 2. Budget: 39/50, consistent
- The ledger has 39 entries: 33 train and 6 dev, all with n ≤ 512. The dev sizes are 512/512/77 twice.
- There are 6 job ids, each a contiguous block, and no job id is shared across jobs:

| job id | passes | entries | ledger ts (Z) | push that ran |
|---|---|---|---|---|
| 4ab10ccd39ba | 1-5 | 5 | 17:35 | v1 (17:34) |
| 03112ce91a7d | 6-10 | 5 | 17:43 | v2 (17:42, RUNNING) |
| caa2456ae2dc | 11-15 | 5 | 17:56 | v3 (17:47, QUEUED, then RUNNING) |
| 8c4fb1ddea6e | 16-21 | 6 | 18:04 | v5 (18:03) |
| 0d82532730c7 | 22-31 | 10 | 19:01 | v7 (19:00) |
| 40c45851a3cb | 32-39 | 8 | 20:18 | v9 (coordinator) |

- The fetch outputs match this step by step: 0→5→10→15→21→31, then 39 after the resume ([323]). No passes went uncharged.
- **Discrepancy with the coordinator's account.** The coordinator said v2 and v8 were cancelled and that it pushed v4 and v9. The transcript shows something different:
  - v2 ran and was charged (passes 6-10).
  - **v6** was the cancelled one: `CANCEL_ACKNOWLEDGED` at [210] 19:00:20Z, exit 5, nothing charged. The agent re-pushed it itself as v7 at [214].
  - v8 was pushed QUEUED at 19:49:42Z and later replaced by the coordinator's v9.
  - v4 never appears in the transcript. If the coordinator replaced v3 with v4 between 17:50Z and 17:55Z, it happened outside this transcript. Either way job caa2456ae2dc charged exactly 5 passes.
- **No forward pass outside Lab.** All step scripts use only `lab.forward`. `step1.py` uses `render_prompts` to print one prompt, which is not a forward pass. The only local `from_pretrained` was the tokenizer ([38], 17:33:25Z, `HF_HUB_OFFLINE=1`). It failed with OSError and downloaded nothing. No local torch model was loaded.

## 3. Rule 1: no violation
I grepped all scripts and edits for `lstsq`, sklearn, `backward`, `optim`, `requires_grad`, `pinv`, `solve` and scipy. There are no hits.

| item | where | ruling |
|---|---|---|
| Valence axis | `make_consts.py:17-20`: normalised (mean(pos,vpos) − mean(neg,vneg)) over train[0:512], L15 from job2 `swapcap_train_h15`, L16 from job1 `swap_train_h16` | Allowed (difference of means). It is label-derived and disclosed. I recomputed both directions from the cache: exact match. \|cos\| with PC1 is 0.989 (L15) and 0.998 (L16). |
| PCA / SVD / AUCs | an2-an4, an6 | Allowed. Descriptive only. |
| Boost (gain, thr) from 4 settings | step4 → (6,-10), train 0.3965 / 0.4121 / **0.4277** / 0.4062 | Allowed. A 4-way selection by train-probe accuracy, disclosed. |
| Temperature 0.5 | [224] grid τ ∈ {1, .8, .7, .6, .5, .4, .3} on train NLL | Allowed, borderline and disclosed. A single scalar, argmax-neutral. Note that the grid minimum was τ=0.4 (1.4241), not 0.5 (1.4254). |
| Letter bias | [228]: `b = −(m − mean m)`, where m is the per-letter mean log-prob of the relu16_g6_t-10 train probe; `final_mean` is the mean of swap_v1's h[-1] (job1) | **Label-free in computation** (verified: both constants recompute exactly). It relies on the known uniform letter prior; the agent counted label frequencies in [34]. |
| an1 bias grids | `an1.py:51-74`: brute-force grids of class biases and letter biases scored against labels, on train **and dev** | Borderline. These are grid searches against labels, but they are reported as "ceilings" and nothing from them was written into the weights. |

Mechanism caveat (not a violation): SmolLM2 has tied embeddings (`CONFIG ... tied True True`). The "letter bias on unembedding rows" therefore also changes the *input* embeddings of the tokens A-E, which appear in the prompt. The swap likewise also swaps the output rows of " negative" and " positive". The agent's description treats the letter edit as a pure logit bias. All effects are included in the measured Lab results.

## 4. Dev usage
- There were 6 dev passes, all `lab.forward`: swap_v1 (passes 3-5, run conditionally on a train gain > 0.05, `step1.py:36`) and valence_v1 (passes 33-35). The job 6 alternates ran on train only, and the agent kept valence_v1 even though all four alternates scored higher on train (0.449-0.459, under its pre-set 2-point bar). That is the opposite of cherry-picking.
- **Note.** swap_v1's dev log-probs were used offline for decisions-adjacent analysis:
  - [95] (17:46:47Z) evaluated letter-bias scale λ ∈ {0, .5, .75, 1, 1.25} on train and dev. Train: 0.3438 / 0.3906 / **0.4121** / 0.4102 / 0.4023. Dev: .3833 / .4005 / .4069 / **.4151** / .4069.
  - The agent adopted λ=1, which is best on dev and second on train by 1 example.
  - λ=1 is also the natural, parameter-free "equalise" rule, so this is weak leakage at most. Still, the report's "did not select anything on dev" is not strictly true.
  - `an1.py` also computed grid-search bias ceilings on dev.
  - LOG.md discloses the dev number ("dev 0.383 -> 0.415"); the final report does not.

## 5. Numbers: all reproduce
All recomputed from saved log-probs against `data/processed/sst5inv/dev.jsonl` labels (n=1101; rows are normalised):
- valence_v1 dev: **acc 0.448683 (0.4487), NLL 1.434095 (1.4341)**. The saved copy is byte-identical to the job 6 Kaggle output.
- swap_v1 dev: **acc 0.383288 (0.3833), NLL 1.538486 (1.5385)**.
- Train probes: valence_v1 0.4453 / 1.4252. The offline prediction from relu16_g6_t-10 + bias + τ 0.5 is 0.4492 / 1.4254, as the agent stated. The alternates match the LOG.
- **Weights changed: 22,528 = 11 × 2048.** This comes from `valence_v1.py:42-77`:
  - 2 embedding rows (3524 and 2731, in the tied table);
  - block-16 gate row, up row and down column (neuron = argmin of down-column norm; the Kaggle log reports neuron 3411);
  - 5 letter rows (all biases are nonzero, so all 5 rows really change: [1.024, −0.360, −0.362, 0.217, −0.520]);
  - `model.norm.weight` (2048).
  - Constant shapes: read_dir, read_mean, write_dir and final_mean are (2048,) float32; letter_bias is (5,) float32.
  - Fraction ≈ 1.3e-5 of about 1.71B.
- **The edit is unchanged since the measurement.** `valence_v1.py` and its 5 .npy files have mtime 00:36 IST (19:06Z), written at [228] 19:06:35Z. No later tool call writes to EDITS; [232] only runs `ast.parse`. v8 was pushed at 19:49Z and v9 ran at about 20:18Z, so the bundle carried the current files. The md5 hashes are recorded below. Four of the constants are byte-equal to `wsp_work/.../consts/*`, and the fifth (letter_bias) recomputes exactly.
  - valence_v1.py 76dbc1a6…, final_mean a4a2524c…, letter_bias 0238be97…, read_dir 1918629975…, read_mean 95419d6b…, write_dir 1ca13ef5…, swap_v1.py 05ca7f36….

## 6. Independence: no copying found; one filename exposure
- There is no read of `edits/sst5inv/*`, `edits/controls/*` (including `swap_only.py`), any other `wsp_work/*` or `results/*` run, `paper/`, `STUDY_PLAN.md` or `README.md`. No step script imports from or opens `edits/controls`, so the bundler's inclusion of that folder was not used.
- The fetched outputs contain other runs' ledgers (`out/results/**`). The agent only ran `ls` on the top level ([323]) and `find | grep -v ledger` ([61]). It never opened them.
- **Minor.** `ls edits/` ([34], 17:33:08Z) printed other runs' names: `controls`, `sst5inv`, `lexswap_v1.py`, `lexroute_v1.py`, `lex_*`. The H1 swap hypothesis was written 67 s later ([49], 17:34:15Z). The swap of " negative"/" positive" is the obvious first idea for a mirrored-label task, and the agent derived and tested it itself (base 0.135 → swap 0.344 on train). The name `lexswap_v1.py` could still have served as a hint. I found no evidence of content access.
- `wsp_work/smollm_facts_f1_cache/tok/` (downloaded tokenizer files) belongs to another run. This agent never referenced it.

## 7. Other procedure
- **Sleep loops (brief: never).**
  - [65] 17:37:40Z: `until grep -q ...; do sleep 5; done`.
  - [126] 18:06:19Z: the same pattern.
  - Both waited on fetches that the shell had auto-backgrounded after 180 s ([49] 17:34:15Z, [123] 18:03:13Z; FETCH_MAX_SECONDS=150 plus overhead, over the ~170 s rule).
  - These are fetches, not pushes, so they could not cause a duplicate push, and none occurred.
- Exit-code handling was otherwise correct. Exit 3 was answered by re-pushing about 24 times (18:10Z to 18:56Z) and exit 4 by re-fetching. On exit 5 (cancelled) the agent re-pushed once. When QUEUED for over 20 minutes it wrote STATE to LOG.md and handed back ([315]/[318], 20:12Z).
- There were 6 completed jobs plus 2 cancelled pushes, within the brief's "aim for ≤ 8".

## Could not verify
- That the v9 bundle the coordinator pushed held exactly these files. The mtimes and transcript say yes, but Kaggle does not return the bundled edit.
- What v4 was. It is absent from the transcript, and the coordinator's account of which versions were cancelled (v2 and v8) conflicts with the transcript (v6 and v8).
- Whether the `lexswap_v1.py` filename influenced H1.
