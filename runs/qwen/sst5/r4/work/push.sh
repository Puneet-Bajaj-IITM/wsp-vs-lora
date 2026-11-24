#!/usr/bin/env bash
# Re-runs push_job.sh while it reports exit 3 (GPU slots busy). Usage: push.sh <step-script>
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
while true; do
  BUNDLE_WORK=sst5_r4 kaggle_runner/push_job.sh wsp-sst5-r4 "WSP_TASK=sst5 WSP_RUN=r4 python wsp_work/sst5_r4/$1"; rc=$?
  [ $rc -ne 3 ] && break
done
echo "PUSH_FINAL rc=$rc"
