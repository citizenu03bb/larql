#!/usr/bin/env bash
# One P1-3 service on Claude's side (docs/phase-b-p1-amendment-a1.md) on GPT-2 XL, under the shared heavy-model
# lock and the limits of AGENTS.md. Same guards as run_service.sh: committed, unchanged protocol, --i-have-approval.
#
#   run_p1_service.sh --step templates --i-have-approval
#   run_p1_service.sh --step solve --fact FACT --i-have-approval
#   run_p1_service.sh --step solve-ref --fact FACT --i-have-approval    # amendment A3
#   run_p1_service.sh --step probe --fact FACT --i-have-approval        # A3 §5, P1-5 probability probe
set -euo pipefail

protocol=docs/phase-b-p1-amendment-a1.md step= fact= approved=
while [ $# -gt 0 ]; do
  case "$1" in
    --step) step=$2; shift 2 ;;
    --fact) fact=$2; shift 2 ;;
    --i-have-approval) approved=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$step" ] || { echo "need --step" >&2; exit 2; }
if [ "$step" = solve-ref ] || [ "$step" = probe ]; then protocol=docs/phase-b-p1-amendment-a3.md; fi
[ -n "$approved" ] || { echo "refused: no --i-have-approval" >&2; exit 3; }
git ls-files --error-unmatch "$protocol" >/dev/null 2>&1 || { echo "refused: $protocol is not committed" >&2; exit 3; }
git diff --quiet HEAD -- "$protocol" || { echo "refused: $protocol differs from HEAD" >&2; exit 3; }

here=$(cd "$(dirname "$0")" && pwd)
venv=${EDIT_CERT_PYTHON:-/mnt/models/finetune/venv/bin/python}
lock="$HOME/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock"
unit="edit-cert-p1-${step}${fact:+-$fact}"

exec flock -w 3600 "$lock" systemd-run --user --wait --collect --unit="$unit" \
  --property=MemoryHigh=22G --property=MemoryMax=24G --property=MemorySwapMax=8G --property=RuntimeMaxSec=1800 \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 --setenv=PYTHONHASHSEED=0 \
  --working-directory="$here" \
  "$venv" -m p1_rome_claude.run "$step" $fact --i-have-approval
