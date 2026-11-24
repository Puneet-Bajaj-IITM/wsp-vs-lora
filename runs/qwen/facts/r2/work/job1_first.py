"""Job 1: first real run of the closed-form circuit. Three scale variants on a train probe, best one on full dev."""
import json

from job_common import WORK, full_dev, probe

OUT = WORK / "out_job1"
OUT.mkdir(exist_ok=True)
VARIANTS = ("hijack_v1", "hijack_w16k", "hijack_c1k", "hijack_late")

results = {}
for edit in VARIANTS:
    summary, diag, stats = probe(edit, OUT)
    results[edit] = {"train_probe": summary, "diag": diag, **stats}
best = max(VARIANTS, key=lambda e: (results[e]["train_probe"]["accuracy"], -results[e]["train_probe"]["nll"]))
results[best]["dev"] = full_dev(best, OUT)
(OUT / "results.json").write_text(json.dumps(results, indent=1))
