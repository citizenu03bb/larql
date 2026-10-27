#!/usr/bin/env bash
# One D1 diagnostic service (docs/phase-b-d1-diagnostic-protocol.md) on the real model, under the shared
# heavy-model lock and the limits of AGENTS.md. Same guards as run_service.sh: committed, unchanged protocol,
# an existing plan, and --i-have-approval.
#
#   run_d1_service.sh --service K [--root DIR] --i-have-approval
set -euo pipefail

protocol=docs/phase-b-d1-diagnostic-protocol.md root="$HOME/Projects/larql-research/phase-b-d1-20261027" service= approved=
while [ $# -gt 0 ]; do
  case "$1" in
    --service) service=$2; shift 2 ;;
    --root) root=$2; shift 2 ;;
    --i-have-approval) approved=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$service" ] || { echo "need --service" >&2; exit 2; }
[ -n "$approved" ] || { echo "refused: no --i-have-approval" >&2; exit 3; }
git ls-files --error-unmatch "$protocol" >/dev/null 2>&1 || { echo "refused: $protocol is not committed" >&2; exit 3; }
git diff --quiet HEAD -- "$protocol" || { echo "refused: $protocol differs from HEAD" >&2; exit 3; }
[ -f "$root/plan.json" ] || { echo "refused: no plan in $root" >&2; exit 3; }

here=$(cd "$(dirname "$0")" && pwd)
venv=${EDIT_CERT_PYTHON:-/mnt/models/finetune/venv/bin/python}
lock="$HOME/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock"
unit="edit-cert-d1-s${service}"

exec flock -w 3600 "$lock" systemd-run --user --wait --collect --unit="$unit" \
  --property=MemoryHigh=22G --property=MemoryMax=24G --property=MemorySwapMax=8G --property=RuntimeMaxSec=1800 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 --setenv=PYTHONHASHSEED=0 \
  --working-directory="$here" \
  "$venv" scripts/phase_b_d1_diagnostic.py service "$service" --root "$(realpath "$root")" --i-have-approval
