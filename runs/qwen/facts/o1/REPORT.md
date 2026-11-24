# Programmer report: qwen facts o1

Final message of the programmer agent, recovered from its transcript.

task: facts
run: o1
edit_name: mem_s2 (edits/facts/o1/mem_s2.py, which loads mem_s2_mem.pt through edits/facts/o1/_memlib.py)
dev_acc: 0.730 (95% CI 0.699 to 0.758; city 0.900, profession 0.663, hobby 0.627). Measured with two lab.forward passes over the full dev set.
dev_nll: 1.163
base_dev_acc: 0.194 (reference given in the brief; base NLL 3.634)
passes_used: 44 of 50 (ledger at results/facts/o1/wsp_ledger.json)
weights_changed: 4,147,200 parameters. These are 900 repurposed neurons in the layer-8 MLP; for each I overwrote its gate_proj row, its up_proj row and its down_proj column (900 × 1536 × 3).
weights_fraction: about 0.27% of the model. That is 10% of the layer-8 MLP.
kaggle_jobs: 6 completed runs (job1, job2, job3 run twice, job4, job5). Job 3 crashed twice on my own bugs, after capture passes had been charged. Job 2 also had several pushes rejected with HTTP 400 before it ran. They were rejected when the bundle was 3.7–11.6 MB; job 1's 0.76 MB bundle went through, so I moved all bulky arrays out of WORK and re-measured keys on Kaggle in every job.

mechanism:
- **Keys.** I used hooks registered only during lab.forward passes to capture the layer-8 MLP input at the last three tokens of each person's name. Each person's key is the mean over their train occurrences (3 or more), minus the mean over all names, and projected orthogonal to that mean.
  - The keys separate people almost perfectly at every layer from 2 to 16 (leave-one-out nearest centroid 0.99–1.00 across templates).
  - Activations at random token positions score at most 0.3, against about 1.0 for the true person.
- **Values.** For each attribute value, the value vector is the mean layer-8 residual at that option's token, minus the mean over the attribute's 10 values. These came from the same capture passes.
- **Oracle tests** (hooks injecting the true value vectors at the name token, 512 train examples):
  - Injection only works at layers 8 and below. Asked-attribute-only at layer 8, scale 1.5, gives 0.93.
  - Writing all three attributes at the one position falls to 0.45–0.58.
  - Writing random vectors of the same norm in place of the other two attributes hurts as much as their real values (0.50). So the cost is the extra norm at a single position, not interference between attribute contents.
- **The fix is to spread the attributes over different name tokens.** City is written at the last name token, hobby at the second-to-last, profession at the third-to-last, each at scale 1.5. The oracle for this layout scored 0.889.
- **The edit itself.** There is one neuron per (person, attribute).
  - gate = 20·(key at that token − 0.6·c(x)). Here c(x) is the projection on the mean MLP-input direction, normalised to about 1 on name tokens. It stands in for the bias that Qwen's MLPs don't have.
  - up = key, so the up-projection output is about 1 for the target person and about 0 otherwise.
  - down = 1.5 × the true value's vector, divided by the gate's saturation value.
  - If a person's key at a given token is nearly identical to another person's (cosine above 0.8), that attribute falls back to the last token. This happened for 25 people at the second-to-last token and 61 at the third-to-last.
  - The 900 neurons overwritten were those with the smallest product of gate, up and down norms.

failed:
- mem_v1, all three values written at the last token (layer 8, scale 1.5): dev 0.508.
- Writing at layers 12 and 16, or at large scale (6), gave nothing on the train oracle set (0.19–0.23).
- mem_g1 used the asked attribute to weight the values, read from a direction at the name token: dev 0.654. That attribute signal transfers poorly to question wordings it wasn't built from (about 0.35 versus 1.0 in-sample).
- mem_s1 used the layout city/profession/hobby at the last/second/third-to-last tokens: dev 0.713. Hobby at the third-to-last token was weak.
- mem_s3 was the same layout with hobby at scale 2: dev 0.690.
- About 13 passes were spent on runs that crashed or had to be redone because of the bundle-size rejections and two bugs.

borderline:
- The down-projection columns are an explicit per-person table derived from train labels: each person's true city, profession and hobby are mapped to that value's option-token mean and written into the weights. Keys and attribute directions are differences of means. No solver, regression or gradients were used.
- I picked mem_s2 out of five edits (mem_v1, mem_g1, mem_s1, mem_s2, mem_s3), all measured on the full dev set. That is mild selection on dev; the gap between the last three (0.69–0.73) is inside the confidence interval.
- I read activations at token positions other than the answer through forward hooks inside charged lab.forward passes. Oracle passes used hooks to edit activations during those same budgeted passes.
- For diagnosis I broke mem_s1's dev accuracy down by which persons had key collisions. That used dev labels for analysis only, and it led to the layout swap tried in mem_s2.
- I never opened the test split, nor any other run's folders.

Files: the log is wsp_work/facts_o1/LOG.md; the scripts are in wsp_work/facts_o1/ (common.py, capture.py, build_edit.py, attrdir.py, job1–job5.py); large arrays are in wsp_work/facts_o1_cache/ (not bundled).
