#!/usr/bin/env bash
# Usage: KAGGLE_USERNAME=<you> kaggle_runner/push_job.sh <job> "<cmd1>" ["<cmd2>" ...]
# Bundles and pushes the private kernel <you>/wsp-<job>, retrying while the account's GPU-session limit is hit,
# then waits until Kaggle reports it queued or running.
# Exit 0: started. Exit 1: bundle failed or is over 1 MB. Exit 3: still blocked after PUSH_MAX_SECONDS (rerun later).
# Exit 6: this job is already queued or running (fetch it instead; PUSH_FORCE=1 overrides).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
PUSH_MAX_SECONDS=${PUSH_MAX_SECONDS:-540}
RETRY_SECONDS=60
START_WAIT_SECONDS=120
ALREADY_ACTIVE=6
JOB="$1"; shift
REF="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}/wsp-$JOB"

# Two concurrent runs of one job would both charge the forward-pass ledger, but only one ledger comes back.
is_active() { case "$("$KAGGLE" kernels status "$REF" 2>&1 || true)" in *QUEUED*|*RUNNING*) return 0 ;; esac; return 1; }
if is_active && [ "${PUSH_FORCE:-0}" != "1" ]; then echo "ALREADY_ACTIVE $REF"; exit $ALREADY_ACTIVE; fi

DIR="$("$ROOT/.venv/bin/python" "$ROOT/kaggle_runner/bundle.py" "$JOB" "$@")" || exit 1
deadline=$((SECONDS + PUSH_MAX_SECONDS))
while true; do
  OUT="$("$KAGGLE" kernels push -p "$DIR" 2>&1)"
  echo "$OUT" | tail -2
  case "$OUT" in *"successfully pushed"*) break ;; esac
  case "$OUT" in *"less than 1 megabytes"*) echo "PUSH_TOO_LARGE bundle over Kaggle's 1 MB limit"; exit 1 ;; esac
  if is_active && [ "${PUSH_FORCE:-0}" != "1" ]; then echo "ALREADY_ACTIVE $REF"; exit $ALREADY_ACTIVE; fi
  if [ $SECONDS -ge $deadline ]; then echo "PUSH_BLOCKED rerun push_job.sh later"; exit 3; fi
  sleep "$RETRY_SECONDS"
done

start_deadline=$((SECONDS + START_WAIT_SECONDS))
while [ $SECONDS -lt $start_deadline ]; do
  if is_active; then echo "STARTED $REF"; exit 0; fi
  sleep 10
done
echo "PUSHED $REF (not yet reported as running; fetch.sh will wait)"
