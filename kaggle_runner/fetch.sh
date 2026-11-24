#!/usr/bin/env bash
# Usage: KAGGLE_USERNAME=<you> kaggle_runner/fetch.sh <job>
# Waits for the kernel <you>/wsp-<job> to finish, downloads its outputs to kaggle_runner/out/<job>/,
# merges the runs' forward-pass ledgers into runs/, and prints the end of the kernel's log.
# Exit 0: downloaded. Exit 4: still running after FETCH_MAX_SECONDS (rerun). Exit 5: cancelled, nothing downloaded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
PYTHON="$ROOT/.venv/bin/python"
POLL_SECONDS=30
DOWNLOAD_RETRIES=12
PAGE_SIZE=200
# LoRA adapters stay on Kaggle (attach them to later jobs with DATASET_SOURCES / KERNEL_SOURCES).
FILE_PATTERN='(results|runs|EXIT_STATUS|\.log$)'
TAIL_LINES=${TAIL_LINES:-40}
FETCH_MAX_SECONDS=${FETCH_MAX_SECONDS:-0}
JOB="$1"
REF="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}/wsp-$JOB"
OUT="$ROOT/kaggle_runner/out/$JOB"

while true; do
  STATUS="$("$KAGGLE" kernels status "$REF" 2>&1 || true)"
  case "$STATUS" in
    *CANCEL*) echo "CANCELLED $STATUS (nothing downloaded; earlier outputs kept)"; exit 5 ;;
    *COMPLETE*|*ERROR*) break ;;
  esac
  if [ "$FETCH_MAX_SECONDS" -gt 0 ] && [ $SECONDS -ge "$FETCH_MAX_SECONDS" ]; then
    echo "STILL_RUNNING $STATUS (rerun fetch.sh $JOB)"; exit 4
  fi
  sleep "$POLL_SECONDS"
done
echo "$(date +%H:%M:%S) $STATUS"

rm -rf "$OUT"; mkdir -p "$OUT"
download_all_pages() {
  local token="" listing
  while true; do
    listing="$("$KAGGLE" kernels output "$REF" -p "$OUT" --page-size "$PAGE_SIZE" --file-pattern "$FILE_PATTERN" \
      ${token:+--page-token "$token"} 2>&1)" || return 1
    echo "$listing" | grep -q "Max retries\|Error" && return 1
    token="$(echo "$listing" | sed -n 's/^Next Page Token = //p' | tail -1)"
    [ -z "$token" ] && return 0
  done
}
for _ in $(seq "$DOWNLOAD_RETRIES"); do
  download_all_pages && [ -f "$OUT/out/EXIT_STATUS" ] && break
  sleep 5
done

"$PYTHON" "$ROOT/kaggle_runner/sync_ledgers.py" "$OUT/out/runs" "$ROOT/runs"
echo "EXIT_STATUS $(cat "$OUT/out/EXIT_STATUS" 2>/dev/null || echo missing)  outputs: $OUT/out"
"$PYTHON" - "$OUT" "$TAIL_LINES" <<'EOF'
import json, pathlib, sys
out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for log in out.glob("*.log"):
    try:
        text = "".join(e.get("data", "") for e in json.loads(log.read_text()))
    except (ValueError, AttributeError):
        text = log.read_text()
    lines = [l for l in text.splitlines() if "Loading weights" not in l and "Warning" not in l]
    print("\n".join(lines[-n:]))
EOF
