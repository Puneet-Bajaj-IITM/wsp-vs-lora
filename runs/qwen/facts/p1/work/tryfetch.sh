#!/bin/bash
cd "$(dirname "$0")/../.."
FETCH_MAX_SECONDS=${W:-110} TAIL_LINES=${T:-60} kaggle_runner/fetch.sh wsp-facts-p1 2>&1 | tail -${T:-60}
echo "rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
