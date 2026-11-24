# Research log: from first observation to final claims

Every experiment in the paper, in the order it was run. Each entry gives the observation that prompted it, the question, what we did, what came out, and what it changed. Numbers are test accuracy unless marked. "WSP" is weight-space programming. "Base" is the unedited model. All Qwen numbers are for Qwen2.5-1.5B-Instruct. SmolLM2 is SmolLM2-1.7B-Instruct.

---

## 0. Starting point

**Observation.** Takehi et al. (2026) showed that a frontier model can read a smaller network's weights and edit them directly, with no gradients. Their headline result fixed Qwen3.5-0.8B's maths reasoning loops by removing one direction from two MLP layers, raising MATH-500 from 7.9% to 48.1%.

**What was missing.** No comparison with ordinary fine-tuning on the same model and task. No measurement of what the edit does to the model's other abilities. No sense of which problems it works on.

**Question.** When does weight-space programming beat fine-tuning, and what does an edit cost the rest of the model?

---

## Phase 1: Pilot on a realistic task

### E1. WSP versus LoRA on predicting real survey answers (Twin-2K)
- **Motivation.** Can a programmer make a small model predict what a real person answered (5 options in random order), and how does that compare with LoRA?
- **Result.**

  | Method | Test acc. |
  |---|---|
  | Base | 30.9% |
  | WSP (Fable, early brief, r1) | 32.8% |
  | LoRA | 42.8% |
  | Simply predicting each question's most common training answer, ignoring the person | 47.1% |

- **Observation.** The programmer mostly remapped option words, i.e. it rearranged what the model already had. No method beat the answer prior.
- **New question.** Is programming limited to rearranging knowledge the model already has? That needs tasks where the amount of existing knowledge is controlled.

---

## Phase 2: The controlled study

### E2. Four tasks that vary how much the model already knows, plus a probe to measure it
- **Motivation.** To test "WSP works in proportion to existing knowledge", we need one model, one answer format, and tasks that step from "all of it" to "none of it".
- **Tasks.**

  | Task | Knowledge in the base model | Base | Linear-probe headroom (probe minus base, dev) |
  |---|---|---|---|
  | SST-5 sentiment | present, read out correctly | 44.9% | +5.6 |
  | SST-5 inverted (mirrored labels) | present, read out backwards | 4.0% | +36.1 |
  | Twin-2K survey answers | population tendencies only | 30.9% | −0.9 |
  | Facts about 300 invented people | absent | 21.1% | +1.6 |

- **Result.** The probe confirms the intended ordering at the two ends: a huge unused signal on SST-5 inverted, nothing on facts.

### E3. One WSP run per task under a shared brief, against LoRA
- **Setup.** Fable programmers r1–r3, 50 budgeted forward passes each, no gradients, no fitted solvers, test never visible, an independent audit afterwards. LoRA uses its default configuration.
- **Observations.**
  - **SST-5 inverted: redirect.** Every programmer independently swapped the embedding rows of " negative" and " positive" (46–48% against LoRA 49%).
  - **Facts: LoRA failed at the default.** It stayed at a uniform guess (22.2%). Higher learning rates also failed.
  - **Facts: WSP succeeded.** Fable hand-built a lookup circuit from the training labels that answered 100% of test questions, in 6–10 passes.
  - **Twin-2K: compile.** Fable compiled the per-question answer prior into the first layers (45.0%, above LoRA's 42.8%).
  - **SST-5: learn.** LoRA 59.0% against WSP 47.3%.
- **Sub-experiment E3b: why did LoRA fail on facts?** A 60-person version showed LoRA can store such facts, but only with about 15 epochs. On the full task, 15 epochs (40,500 examples, about 3.6 T4-hours) reached 98.8%.
- **What changed.** The hypothesis was half wrong. Programming *can* add knowledge, by compiling labels into an explicit table and machinery to look it up.
- **New question.** What does such a circuit do to everything else?

### E4. Side-effect checks: option rotation and another task
- **Motivation.** A circuit that recalls 900 invented facts might break the model, and target-task accuracy would never show it.
- **Result.**
  - **Fable's circuits** for facts and Twin-2K drop SST-5 accuracy from 42.3% to 18.7–19.9%, below chance.
  - **LoRA and Opus's facts edit** leave SST-5 intact.
- **Concept introduced.** *Bypass* edits overwrite early-layer computation with a hand-built circuit. *Through* edits work through the model's existing pathways.
- **New question.** Is this about the method, or about the programmer?

### E5. A second programmer model (Opus 5.5, run o1 per task)
- **Motivation.** Separate the method from the programmer.
- **Result.**
  - **SST-5 inverted.** Opus found the same swap (45.7%).
  - **Facts.** Opus built a ROME-like memory in layer-8 MLP neurons: 76.9%, with SST-5 intact.
  - **Twin-2K.** Opus put a train-derived offset into one neuron (31.9%, intact).
  - **SST-5.** No Opus edit beat base.
- **What changed.** The programmer matters when an edit must be *built*, and different programmers choose bypass or through.

---

## Phase 3: Strengthening the claims

Each experiment here answers a specific weakness of phase 2.

### E6. Broad side-effect evaluation (WikiText-2 perplexity and 500 MMLU questions)
- **Weakness addressed.** One proxy task (SST-5) is not "general ability".
- **First result.**
  - **Bypass edits destroy the model.** Fable facts r1 drives MMLU to 22.2% (base 59.6%), and its WikiText loss overflows. Twin r2 gives MMLU 23.4% and next-token top-1 0.1%.
  - **Through edits look damaged too.** Their perplexity rises from 15.9 to between 1,800 and 6,300, yet MMLU stays at 55–60%.
- **E6b: is that perplexity real damage?** Nearly every through edit includes a global temperature change. We added next-token top-1 accuracy, which a temperature change cannot affect, and the best loss reachable with one oracle logit rescale.
- **Result.** Through edits keep top-1 at base (45.5–45.8% against 45.9%), and one rescale recovers base loss. Bypass edits stay at 0% top-1, with no rescale recovering them.
- **What changed.** The damage finding now rests on standard benchmarks. Raw perplexity is a misleading side-effect metric for edits that carry a temperature: a methodological lesson in itself.

### E7. Controls and statistics
- **Weakness addressed.** Is the SST-5-inverted swap effect semantic, or just "any edit disrupts the model"? Are the differences significant?
- **Result.**
  - **Unrelated-word swap** (" table"/" river"): 4.0%, identical to base.
  - **The " negative"/" positive" swap alone**, without temperature: 46.2%, essentially the full edit (46.4%).
  - **Temperature-only control on the base model:** NLL improves, accuracy is unchanged.
  - **Paired bootstrap:**

    | Comparison | Difference | Significance |
    |---|---|---|
    | Best SST-5-inverted run vs LoRA | not significantly different | p = 0.37 |
    | Facts circuit vs LoRA | +1.2 | significant |
    | Twin r2 vs LoRA | +2.3 | p = 0.008 |
    | SST-5 WSP vs LoRA | −11.7 | significant |

- **What changed.** The swap does all the work, and the comparisons are statistically grounded.

### E8. LoRA learning curves (256 and 1,024 distinct training examples, same number of steps)
- **Weakness addressed.** Each WSP run reads the labels of about 2,000–3,000 training examples. Is LoRA's win on SST-5 just more data?
- **Result.**

  | | 256 examples | 1,024 examples | Full pool | WSP |
  |---|---|---|---|---|
  | SST-5 | 54.2% | 57.8% | 59.0% | 46.7–48.0% |
  | SST-5 inverted | 29.1% | 53.1% | 49.3% | 45.7–48.1% |

- **What changed.**
  - **SST-5:** LoRA wins even with a tenth of the data, so WSP's shortfall is not a data effect.
  - **SST-5 inverted:** WSP is the better choice when labels are scarce.

### E9. More LoRA seeds (seeds 1–2 for SST-5 and SST-5 inverted)
- **Weakness addressed.** Every LoRA number was a single seed.
- **Result.**
  - **SST-5 is stable:** 56.6–59.0%.
  - **SST-5 inverted: only 1 of 3 seeds learns** (49.3%; the others stay at 23.9% and 27.7%). Training starts confidently wrong (loss 3.9) and sits on the uniform-guess plateau (loss ln 5 ≈ 1.61). Seed 0 escapes only after about 200 of 500 steps.
  - **Not a configuration artefact:** re-running seed 0 with the same flag change reproduces 49.5%.
- **What changed.** On a task the model knows but reads out backwards, default LoRA is unreliable. Programming found the fix every time it was tried.

### E10. The "preserve" condition
- **Weakness addressed.** The phase-2 brief never asked programmers to protect other abilities. Can a programmer compile knowledge without breaking the model if told to? A held-out SST-5 set was provided, with 40% as the acceptance bar.
- **Twin-2K (p1): yes, partly.**
  - Fable switched from a bypass circuit to a gated, rank-3 change to four attention heads.
  - Test 35.2% (base 30.9%).
  - SST-5 42.2% against base 42.3%; 41.4% on the preserve examples it never saw.
  - WikiText unchanged, MMLU −2.
- **Facts (p1): no.**
  - The best edit gained about 4 points on dev examples it had not selected on.
  - It fell below the bar (SST-5 38.1%).
  - It did real but partial damage: MMLU 55.6%, top-1 39.7%.
- **What changed.** The brief decides the edit type. Preserving the model costs most of the compiled gain, and on facts the programmer could not find an edit that both added knowledge and preserved the model.

### E11. Replications
- **Weakness addressed.** Few runs per task.
- **SST-5: three Fable runs, three mechanisms, one ceiling.**
  - r1: hand-built circuit, 47.3%.
  - r4: one MLP neuron, 48.0%.
  - r5: attention-bias steering, 46.7%.
  - All three fix the same readout bias: the model almost never says "neutral" and over-uses "very".
- **Facts r4: a third bypass circuit.** 100% test, and it destroys the model (MMLU 25.8%, top-1 0%, 94% of SST-5 answers are "E").
- **Twin r4: a through edit, but excluded.**
  - **What it built.** Fable re-weighted the option-reading attention keys in layers 18/20/21: 37.0% test (base 30.9%), with the model intact (MMLU 58.8%, top-1 45.6%).
  - **Why it is excluded.** The audit found a rule-1 violation: one discarded pass used a least-squares read-out fitted to the label table, and the final read-out is mathematically a ridge regression on the labels. It goes in the appendix as a protocol deviation.
  - **Sensitivity.** Including it changes no conclusion. It would be the best intact Twin edit, still below LoRA and the answer prior.
- **What changed.** The regime results and the bypass pattern reproduce.

### E12. A second base model (SmolLM2-1.7B-Instruct)
- **Weakness addressed.** One base model.
- **Baseline and probe.** The probe shows the same knowledge map: headroom +25 on SST-5 inverted, +2 on SST-5, +2 on facts.
- **LoRA** on SST-5 inverted gets stuck on the same plateau (22.7%).
- **SST-5 inverted (Fable f1).**
  - The programmer independently found the same " negative"/" positive" swap: 11.4% → 38.6%. The final edit reaches 45.4%.
  - The unrelated-word control does nothing.
  - MMLU and top-1 are unchanged.
- **Facts (Fable f1).** Again a layer-0/1 bypass circuit: 100% test, and again it destroys the model (MMLU 22.6% against 48.8% base, top-1 0%).
- **What changed.** Both the redirect result and the compile-by-bypass result hold on a second model family.

### E13. Programming plus fine-tuning
- **Weakness addressed.** Are the methods rivals or complements? Can an edit be a starting point for LoRA?
- **Result** (3 seeds each):

  | Task | Edit, then LoRA (4,000 examples) | LoRA alone, same budget | Edit alone |
  |---|---|---|---|
  | Facts (Opus o1 edit) | 99.0–99.7% | 22.2% (LoRA needed 40,500 examples for 98.8%) | 76.9% |
  | SST-5 inverted (Fable r1 swap) | 57.3–58.5% | 23.9–49.3% | 46.4% |

- **What changed.** This is the most useful practical result. A programmed edit gives fine-tuning a working structure to refine. That fixes both LoRA failure modes we saw: slow storage of new facts, and the plateau on a confidently wrong start.

### E14. Cost
- **API cost.** About $16–$86 per finished programmer run at list prices; outliers are facts p1 at $364 and Twin p1 at $103. Audits cost about $1–$14 each. All programmers together: about $1,350 (including stopped runs).
- **GPU time.** WSP: minutes, at most 50 passes of 512 examples. LoRA: 29 minutes per SST-5 task, 3.6 h for facts, 5.7 h for Twin-2K.
- **Kaggle.** The free GPU quota covered all compute.

### E15. Audits and protocol incidents
- **Audit outcome.** Every finished run was independently audited. All but one are "compliant with notes".
- **Recurring notes:**
  - label-derived tables (disclosed);
  - selection on dev in some runs;
  - sleep-poll loops;
  - file-name exposure through directory listings;
  - one random-weight local stand-in test;
  - undisclosed use of dev activations (facts p1);
  - probable duplicate job runs with uncharged passes (facts p1, job 4). This led to harness fixes: a guard against concurrent jobs, and job IDs on ledger entries.
- **One violation.** Twin r4 is excluded from the main comparison for a rule-1 violation (a label-fitted regression read-out) and reported in full in the appendix. This follows standard practice: main results use protocol-compliant runs, and deviations are reported, not dropped.
- **Orchestrator errors, all logged:**
  - one run (SST-5 r5) was told another run's result mid-run; the audit showed its design predated the message;
  - kernel deletions removed some remote copies; local copies were kept.

### E16. Controls on the combination (SST-5 inverted)
- **Weakness addressed.** E13 compared the combination only with default LoRA. Three questions a reviewer would ask: does tuned LoRA alone get there, is the programmer's edit needed or would any swap do, and does it hold on the second model?
- **Result** (3 seeds each, test accuracy):

  | Setup | Seeds | Mean |
  |---|---|---|
  | Qwen, LoRA alone, 4,000 examples, lr 3e-4 | 19.4 / 19.5 / 20.0% | 19.6% |
  | Qwen, LoRA alone, 12,000 examples | 60.5 / 61.2 / 58.1% | 60.0% |
  | Qwen, hand-written label swap (46.2% alone), then LoRA | 58.5 / 57.6 / 58.0% | 58.0% |
  | SmolLM2, LoRA alone | 22.7 / 21.5 / 22.3% | 22.2% |
  | SmolLM2, Fable f1 edit, then LoRA | 55.9 / 56.6 / 56.1% | 56.2% |

- **Training curves.** Every LoRA run sits at the uniform-guess loss (about 1.6) before it learns. With 12,000 examples all three seeds left it between batches 250 and 750 of 1,500; with 4,000 only seed 0 did. A larger learning rate kept all three on it.
- **Side effects (SmolLM2 combination).** MMLU 46.6–48.0% (base 48.8%), top-1 46.0% (base 46.0%).
- **What changed.** The combination claim narrows. The edit lets LoRA learn on every seed of both models at 4,000 examples, but LoRA alone matches it with three times the data, so the edit saves data rather than raising the ceiling. The useful part is the two-row swap, which can be written by hand; the programmer's further changes add nothing after LoRA.

---

## So what: why these findings matter

1. **A map of when to program and when to fine-tune.** It holds on two model families:

   | Regime | Situation | Better tool |
   |---|---|---|
   | Redirect | The model knows the answer but reads it out wrongly | Programming finds tiny, readable, reproducible fixes. It is more reliable than default LoRA and better with few labels. |
   | Compile | The answer is a table that can be written down from labels | Programming matches or beats LoRA, but its most accurate route bypasses the network. |
   | Learn | The improvement needs a better mapping | LoRA wins clearly, even with a tenth of the data. |

2. **Task accuracy alone is a misleading way to judge an edit.** The most accurate programmed edits disable the rest of the model, on two model families. Raw perplexity is misleading in the other direction: it flags harmless temperature changes as damage. We propose a minimal side-effect protocol:
   - next-token top-1 accuracy;
   - a general-knowledge benchmark;
   - an unrelated task;
   - option-order consistency.
3. **The instruction decides the edit type.** Asked to preserve the model, the same programmer moved from a bypass circuit to an edit through the model's own pathways. Preservation has a price: most of the compiled gain on Twin-2K, and all of it on facts. Any brief for a programmer should state what must be preserved, and any evaluation should check it.
4. **Programming and fine-tuning are complements.** A programmed edit makes a good starting point for LoRA. It turned a fact-storage problem that took LoRA about 4 GPU-hours into a 4,000-example run at 99%+, and it removed LoRA's plateau on SST-5 inverted on both models. LoRA alone gets there with about ten times (facts) or three times (SST-5 inverted) the data, so the edit saves data rather than raising the ceiling. This suggests a workflow: program first (minutes, readable), then fine-tune.
5. **Programming behaves like a compiler, not a learner.** In every case where it beat fine-tuning, the programmer wrote an explicit object into the weights: a swapped word pair, a fact table, an answer-rate table. That explains both its strengths and its ceiling.
