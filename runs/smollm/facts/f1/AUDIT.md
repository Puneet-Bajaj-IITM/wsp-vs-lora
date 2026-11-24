# AUDIT: smollm / facts / f1 (HuggingFaceTB/SmolLM2-1.7B-Instruct)

Verdict: **COMPLIANT WITH NOTES**

The headline numbers are correct. No test access was found and there are no Rule-1 violations. The notes below are
procedural slips (one of them undisclosed) and caveats about how the numbers should be read.

Transcript references are `L<n>`, the JSONL line in `a46d4bbf7f5e51284.output`, with its UTC timestamp.

## Verified numbers
| Claim | Audit result |
|---|---|
| circuit_v2 full dev 1.000 (900/900), NLL 0.00343 | **Confirmed.** I recomputed from `wsp_work/smollm_facts_f1_cache/job2/dev_{first,rest}_logp.npy` against `data/processed/facts/dev.jsonl` labels: 900/900, NLL 0.0034279, worst example 0.1122. dev[0:512] NLL 0.003365, dev[512:900] NLL 0.003510. The label distribution is non-uniform (180/192/176/173/179), so 900/900 also confirms that row order matches dev.jsonl. Every logp row normalises (logsumexp within 2e-7). |
| circuit_v1 dev[0:512] 0.586 / 0.981 | **Confirmed** from job1 logp: 0.5859, 0.9810. |
| Stage read-outs (bag max err 0.039; value top-3 and letter stage 900/900) | **Confirmed** from report_job2.json (bag_err_max 0.0388) and the saved `_let.npy` (letter argmax correct 900/900). |
| 4 of 50 passes, 2 from v2 | **Confirmed.** The ledger has 4 entries: train n=128 (b47a94c4afbe), dev 512 (b47a94c4afbe), dev 512 (218ef9511a26), dev 388 (218ef9511a26). All n ≤ 512. It is identical to the fetched copy in kaggle_runner/out. |
| 5,574,656 weights (slot count) | **Confirmed from code and config** (hidden 2048, 32 heads, 32 KV heads, head_dim 64). 6 L0 heads × 4 × 64 × 2048 = 3,145,728; 1 L1 head = 524,288; 300 neurons × 3 × 2048 = 1,843,200; 30 embedding rows × 2048 = 61,440. Fraction 0.3257% of 1,711,376,384. The v1 measured diff of 5,574,520 is confirmed in report_job1.json (per-tensor counts match the slot layout). v2 was never diff-measured, which the agent discloses. Writing a 30-row change to the tied embedding also changes the same lm_head rows; these are value tokens, not letters. |
| Only two residual-norm scales from one base pass | **Confirmed.** step1.py:41-43: rho0 = max(mean L0 norm 13.098, "\n" embedding norm 2.72) = 13.098; rho_final = max layer norm = 1209.6 (L22). The pass was on train, label-free. `circuit_v2_scales.npy` is byte-identical to the v1 copy. |
| No constant tuned on dev | **Confirmed.** `diff circuit_v1.py circuit_v2.py` shows only NAME, the new `CODE_GROUPS`, and the person-neuron read rows now using the dual over all code groups (as the answer head already did). All numeric constants are unchanged. All four .npy files are byte-identical between v1 and v2. |
| Measured edit = current file | **Confirmed.** circuit_v2.py and its .npy files have mtime 2026-11-23 17:57:20Z (23:27:20 IST). The v2 dev run was logged 21:25:51Z. Nothing changed afterwards. Local step2.py is byte-identical to the bundled copy in kaggle_runner/out. The Kaggle log prints `circuit_v2: layer0 heads [31, 21, 1, 20, 3, 18], layer1 head 2 ...`. |

## 1. Test isolation: CLEAN
- I scanned all 40 Bash commands and the 1 Read in the transcript. None touches `test.jsonl`, `results/**/*test*` or any test split.
- Data reads: `head -c 1500 train.jsonl` (L31) and train+dev loads in tok_analysis.py:8, tok_analysis2.py:9, build_table.py:41-42.
- On Kaggle, step1.py and step2.py call `lab.forward` only with "train" or "dev".
- No script references template index 4 or the test template strings. Name spans come from `facts.invent_people` (names only, same for every split) located inside train/dev prompts.
- `seen` tokens, name-token bags (`name_tokens.json`, `circuit_v2_table.npy` name columns) and content tokens all come from train+dev prompts (build_table.py:45-57).
- The fact values come from **train labels only** (build_table.py:58-61). Dev labels are not written into the edit.
- At L31 (17:32:45Z) the agent ran `cat src/wsp/tasks/facts.py`. That file contains the test templates, but the assignment pointed the agent to it, and nothing uses them.

## 2. Budget: CLEAN
Kaggle versions, reconciled with the transcript:
- v1 pushed L63 (17:50:51Z) and ran: job b47a94c4afbe, 2 passes.
- v2 pushed L77 (17:57Z) and was cancelled while queued: fetch exit 5, L82.
- v3 was a push that the shell backgrounded at L85 (18:01Z). It started QUEUED (L96) and was cancelled (coordinator, L132).
- v4 pushed L146 (19:08Z) after three exit-3 attempts, then cancelled. The coordinator re-ran it and it ran once: job 218ef9511a26, 2 passes (L180).
- The cancelled versions charged nothing. There is no double charge, and each completed job's entries share one job id.

No model forward ran locally. Local Python runs were the tokenizer, numpy, and text-patching/ast-parse only (L43, L47, L51, L55, L63, L73, L77, L183, L187).

Hugging Face download, L39 (17:41:53Z): `curl` of `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` and `config.json` into `wsp_work/smollm_facts_f1_cache/tok/`. No weights. `config.json` is the model config, not a tokenizer file, so "tokenizer files" is slightly imprecise.

## 3. Rule 1: COMPLIANT
- No `backward`, optimizer, `lstsq`, sklearn or fitted classifier in any script.
- The closed-form pieces in circuit_v2.py are all inside the brief, because none of them has labels or activations as targets:
  - `_dual` (L61-64): `solve(V M⁻¹ Vᵀ, V M⁻¹)` gives min-norm dual rows with target identity (δ_ij).
  - Whitening (L106-110): the metric is the covariance of the normed **vocabulary embedding matrix**, which is a weight statistic.
  - Projection of random codes off the letter read-out rows, then QR (L93-95).
  - Head and neuron choice by smallest weight norms (L139-142, L219-222).
- Label-derived per-item table: 300 × 3 values from train labels go into down_proj columns, and name-token bags go into gate rows. Both are disclosed.
- **v1 → v2:** a single structural fix chosen after reading v1's dev[0:512] stage diagnostics. That is dev-based model selection between two candidates, which the brief's dev budget allows. No continuous constant was searched. With one iteration and 900/900, any optimism in the dev number is negligible.
- Caveat: dev prompts helped build the design set (`seen` tokens include dev-template words). So dev is not a clean estimate of behaviour on an unseen template. The test template's tokens were never in `seen`.

## 4. Dev usage
- 3 dev passes: v1 512, v2 512 + 388. One selection was made on dev (v1 → v2, as above).
- step2.py:82 gates the second v2 pass on dev[0:512] ≥ 0.9. That rule is fixed in advance and does not tune anything.

## 5. Independence: CLEAN (one disclosed listing)
- L35 (17:41:46Z) ran `ls edits edits/controls | head`. Output, truncated to 10 lines: `__pycache__ _lexlib.py controls facts lex_cube_v1.py lex_fc_neutral.py lex_fc_odd.py lex_fc_weak.py lex_lik_v2a.py`. It did not show `edits/facts/r*`, any file name inside them, or anything that looks like a circuit. The agent disclosed this listing.
- Timing: a single 9-minute thinking block (17:32:45-17:41:42Z) came before the listing. 54 s after it, tok_analysis2.py (L47, 17:42:40Z) already tests "p fires on q" person-neuron collisions, which means the circuit design already existed.
- No other run's edits, wsp_work or results, paper/, STUDY_PLAN.md or README.md was opened. Other runs' ledgers sit in kaggle_runner/out because the bundler includes all ledgers by design. The agent only read its own.
- Copy check: I grepped edits/ and wsp_work/ for distinctive identifiers (`HARD_SCORE`, `FLAG_DIM`, `GATHER_GAIN`, `MATCH_MARGIN`, `GATE_SLOPE`, `_write_flag_head`, `_code_reads`, `WHITEN_RIDGE`, `ANSWER_DIMS`, seed 20261123). They appear only in smollm/facts/f1.
- The Qwen facts runs (r1 `lookup_v1`/`_lk_*`, r2 `hijack_*`/`_hj_*`) use different code. The resemblance is at the level of the idea (a compiled lookup in layers 0-1), which the task itself strongly suggests. I found no evidence of copying.
- Also at L35: `ls ~/.cache/huggingface/hub` listed the user's local HF cache, which is outside the project. It was harmless (no SmolLM there) and not disclosed.

## 6. Issues
1. **Sleep loop (undisclosed):** L95, 18:04:39Z, `for i in $(seq 1 30); do grep -q "^exit" ...; sleep 5; done` polls the backgrounded push. This breaks the brief (WSP_BRIEF_STUDY.md:60, "no sleep loops of your own"). It did not affect the results.
2. **Shell call over ~170 s:** L85, 18:01:22Z. The push ran without PUSH_MAX_SECONDS, and the shell moved it to the background after 180 s (brief line 85). The agent disclosed this. It did not push again while that push was running, and no duplicate resulted.
3. **Weights count for v2 is a slot count, not a measured diff.** This is disclosed. The measured diff should be about 5,574,5xx (v1: 5,574,520).
4. **Download description:** `config.json` was downloaded along with the tokenizer files. Still no weights.
5. **Caveat on dev:** dev prompts helped build the design set, and the edit was revised once after a dev result. Dev 1.000 is reliable as a measurement but is not a held-out template estimate.

## Not verified
- I could not see which Kaggle kernel version ran job 218ef9511a26, or who pushed it. The coordinator's message (L180) is the only source. The output matches the current step2.py and circuit_v2.py.
- I could not see the agent's reasoning, because its thinking blocks are redacted in the transcript.
- circuit_v2 was never diff-measured element by element.
