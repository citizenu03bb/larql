#!/usr/bin/env bash
# One certification service on a real model, under the shared heavy-model lock and the memory limits of
# AGENTS.md. Refuses unless the protocol file is committed and unchanged, and unless --i-have-approval is given
# (a model forward needs the author's go-ahead for that protocol).
#
#   run_service.sh --protocol docs/<protocol>.md --run RUN_DIR --service K [--model-root PATH] --i-have-approval
set -euo pipefail

protocol= run= service= model_root=/mnt/models/hf-models approved=
while [ $# -gt 0 ]; do
  case "$1" in
    --protocol) protocol=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;
    --service) service=$2; shift 2 ;;
    --model-root) model_root=$2; shift 2 ;;
    --i-have-approval) approved=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$protocol" ] && [ -n "$run" ] && [ -n "$service" ] || { echo "need --protocol, --run, --service" >&2; exit 2; }
[ -n "$approved" ] || { echo "refused: no --i-have-approval" >&2; exit 3; }
git ls-files --error-unmatch "$protocol" >/dev/null 2>&1 || { echo "refused: $protocol is not committed" >&2; exit 3; }
git diff --quiet HEAD -- "$protocol" || { echo "refused: $protocol differs from HEAD" >&2; exit 3; }
[ -f "$run/plan.json" ] || { echo "refused: no plan in $run" >&2; exit 3; }

here=$(cd "$(dirname "$0")" && pwd)
venv=${EDIT_CERT_PYTHON:-/mnt/models/finetune/venv/bin/python}
lock="$HOME/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock"
unit="edit-cert-$(basename "$run")-s${service}"

exec flock -w 3600 "$lock" systemd-run --user --wait --collect --unit="$unit" \
  --property=MemoryHigh=22G --property=MemoryMax=24G --property=MemorySwapMax=8G --property=RuntimeMaxSec=1800 \
  --working-directory="$here" \
  "$venv" -m edit_cert run --run "$(realpath "$run")" --service "$service" --model-root "$model_root"
