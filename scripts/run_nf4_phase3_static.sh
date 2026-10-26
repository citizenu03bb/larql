#!/usr/bin/env bash
# Phase 3 driver: one bounded service per group, each under the shared heavy-model lock. Stops at the first failure.
set -euo pipefail
ROOT=${1:?trace root}
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=/home/pat/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock
PY=/mnt/models/finetune/venv/bin/python
for g in calib T1 T2 T3 T4 T5 R1T6; do
  if [ "$g" != calib ] && ! "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$ROOT/calib/summary.json'))['calibration']['go'] else 1)"; then
    echo "calibration stop: no test services"; exit 0
  fi
  flock -w 3600 "$LOCK" systemd-run --user --wait --pipe --collect --unit="larql-nf4-phase3-$g-20261026" \
    --property=MemoryHigh=22G --property=MemoryMax=24G --property=MemorySwapMax=8G --property=RuntimeMaxSec=1800 \
    --working-directory="$HERE" "$PY" -u run_nf4_phase3_static.py "$ROOT" --group "$g" 2>&1 | grep -v 'Loading weights' | grep -v '^{"ordinal"' || true
  "$PY" -c "import json,sys;s=json.load(open('$ROOT/$g/summary.json'));print('$g',s['status'],s.get('forwards'));sys.exit(0 if s['status']=='complete' else 1)"
done
"$PY" run_nf4_phase3_static.py "$ROOT" --group read > /dev/null
"$PY" check_nf4_phase3_static.py "$ROOT" | tail -40
