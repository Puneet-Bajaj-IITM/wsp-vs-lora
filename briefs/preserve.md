# Addendum for "preserve" runs

Everything in `WSP_BRIEF_STUDY.md` applies unchanged. This addendum adds one requirement.

## The requirement
Your edited model must keep the base model's general abilities. Improving the target task matters, but
an edit that breaks the model for other uses does not count as a success.

You can measure one other ability during your run. `lab.examples("preserve")` returns the SST-5
sentiment dev set (1,101 examples). `lab.forward("preserve", idx, note=...)` scores your current model on
it. That forward is charged to the same 50-pass ledger, with the same 512-example cap. The unedited base
model scores about 42.3% there.

**Acceptance criterion:** your final edit must score at least 40.0% on the full preserve set, which costs
3 passes, and you must report that number. If no edit meets both the criterion and a gain on your task,
report the best one that meets the criterion.

The orchestrator will also check your final edit on held-out general tests you cannot see: text
perplexity and general-knowledge multiple choice. So do not tune only to the preserve set; aim for an
edit that leaves the model's normal computation intact.

## Report
Add to your report: the full-preserve-set accuracy of your final edit, and a sentence on how your edit
avoids interfering with the model's other computation.
