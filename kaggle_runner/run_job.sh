#!/usr/bin/env bash
# Usage: KAGGLE_USERNAME=<you> kaggle_runner/run_job.sh <job> "<cmd1>" ["<cmd2>" ...]
# Bundles the project, pushes a private GPU kernel (waiting out the GPU-session limit), then
# waits and downloads via fetch.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUSH_BLOCKED=3
JOB="$1"; shift
while true; do
  PUSH_MAX_SECONDS=3600 "$ROOT/kaggle_runner/push_job.sh" "$JOB" "$@"; rc=$?
  [ $rc -eq 0 ] && break
  [ $rc -ne $PUSH_BLOCKED ] && exit $rc
done
exec "$ROOT/kaggle_runner/fetch.sh" "$JOB"
