#!/usr/bin/env bash
# Usage: wsp_work/twin_p1/push.sh <script.py>   (short retry window so the call stays in the foreground)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BUNDLE_EXCLUDE="edits/facts/*,edits/facts/*/*,edits/facts/*/*/*" PUSH_MAX_SECONDS=${PUSH_MAX_SECONDS:-50} BUNDLE_WORK=twin_p1 \
  kaggle_runner/push_job.sh wsp-twin-p1 "WSP_TASK=twin WSP_RUN=p1 python wsp_work/twin_p1/$1" > wsp_work/twin_p1_cache/push.out 2>&1
rc=$?
tail -2 wsp_work/twin_p1_cache/push.out
echo "rc=$rc"
