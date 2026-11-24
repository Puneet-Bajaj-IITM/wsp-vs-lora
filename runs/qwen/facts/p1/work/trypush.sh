#!/bin/bash
# one bounded push attempt window (~125s) for the given python script name in WORK
cd "$(dirname "$0")/../.."
BUNDLE_EXCLUDE="*.pt,o1/*.npy,r1/*.npy,r2/*.npy,r3/*.npy,r4/*.npy,r5/*.npy,controls/*.npy,edits/p1/*.npy" BUNDLE_WORK=facts_p1 PUSH_MAX_SECONDS=${W:-125} kaggle_runner/push_job.sh wsp-facts-p1 "WSP_TASK=facts WSP_RUN=p1 python wsp_work/facts_p1/$1" | tail -4
echo "rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
