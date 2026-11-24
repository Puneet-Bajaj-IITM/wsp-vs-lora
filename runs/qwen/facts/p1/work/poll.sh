#!/bin/bash
# poll.sh <output-file> : wait up to ~160s for a push/fetch terminal marker
F="$1"
for i in $(seq 16); do
  grep -qE "STARTED|PUSHED|PUSH_BLOCKED|EXIT_STATUS|STILL_RUNNING|exited with code" "$F" && break
  /bin/sleep 10
done
tail -${2:-3} "$F"
