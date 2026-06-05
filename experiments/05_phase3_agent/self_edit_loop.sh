#!/usr/bin/env bash
# self_edit_loop.sh — the self-editing constellation loop (Phase 3 "THE GOAL" artifact).
#
# Demonstrates a model editing its OWN weights and verifying the result, end to end:
#   reset → /edit each fact (per-slot τ auto-calibration)
#         → /recalibrate (co-active τ over the whole constellation)
#         → /reorthogonalize (kill emergent cross-fact leak)
#         → verify (every target pushes AND every held-out stays native)
#         → verdict → reset
#
# This is the deterministic reference the Pi self-editing agent drives via the larql-edit skill.
# Run it directly to confirm the loop is green before a live agent demo.
#
# Port: $LARQL_PORT (default 8770; 8765 = Aftertone TTS, never use). Needs the daemon running.
set -euo pipefail
H="http://127.0.0.1:${LARQL_PORT:-8770}"

# --- constellation under edit ------------------------------------------------
# Shared base contrast set (real + fictional, mixed categories per the QR-gate rule).
BASE='"The capital of Olympus is","The capital of France is","The capital of Japan is","The capital of Italy is","The capital of Brazil is"'
# facts: "PROMPT-ENTITY<TAB>TARGET-TOKEN"
# Default = the validated smoke_reorthogonalize.sh pair (Egypt→Superman is the canonical
# emergent leak this loop repairs). Pick targets whose FIRST BPE is multi-char and distinctive:
# single-letter first tokens (e.g. "Odin"→"O", "Sauron"→"S") fire promiscuously and poison the
# constellation — they dominate the down-vector superposition and trip prefix-match leak checks.
FACTS=(
  $'Atlantis\tPoseidon'
  $'Krypton\tSuperman'
)
# held-outs: real entities NOT in the contrast set — must stay native after editing.
HELDOUTS=(Spain Egypt Germany Belgium)

prompt_of() { echo "The capital of $1 is"; }

req() { # req METHOD PATH [DATA]
  # 900s budget: /reorthogonalize re-gates + re-calibrates τ over the whole
  # constellation (N facts × grid × infers) and runs well past a single edit.
  local m="$1" p="$2" d="${3:-}"
  if [ -n "$d" ]; then
    curl -sS --max-time 1500 -X "$m" "$H$p" -H 'Content-Type: application/json' -d "$d"
  else
    curl -sS --max-time 1500 -X "$m" "$H$p"
  fi
}

top1() { # top1 "<prompt>"  -> "TOKEN<TAB>PROB"
  req POST /infer "$(jq -n --arg p "$1" '{prompt:$p, top_k:5}')" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); p=d['predictions'][0]; print(p['token']+'\t'+str(p['prob']))"
}

echo "=== Phase 3 self-editing loop @ $H ==="
req GET /health >/dev/null || { echo "daemon not reachable on $H — start it first" >&2; exit 1; }

echo "--- 0. reset ---"
req POST /reset | jq -c .

echo "--- 1. baseline (pre-edit) ---"
for f in "${FACTS[@]}"; do
  ent="${f%%$'\t'*}"; IFS=$'\t' read -r t1 p1 <<<"$(top1 "$(prompt_of "$ent")")"
  printf '    %-10s -> %s (%.0f%%)\n' "$ent" "$t1" "$(python3 -c "print($p1*100)")"
done

echo "--- 2. /edit each fact (per-slot τ auto-calibration) ---"
for f in "${FACTS[@]}"; do
  ent="${f%%$'\t'*}"; tok="${f##*$'\t'}"
  payload=$(jq -n --arg p "$(prompt_of "$ent")" --arg t "$tok" \
             --argjson c "[$BASE]" '{target_prompt:$p, target_token:$t, contrast_prompts:$c}')
  tau=$(req POST /edit "$payload" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['record']['tau'])")
  printf '    %-10s -> %-9s  τ=%s\n' "$ent" "$tok" "$tau"
done

echo "--- 3. /recalibrate (co-active τ over the constellation) ---"
req POST /recalibrate | python3 -c "import sys,json; d=json.load(sys.stdin); print('    recalibrated',d['recalibrated']); [print(f\"      {c['target_token']!r}: τ {c['tau_old']}->{c['tau_new']}\") for c in d.get('changes',[])]"

is_push() { # is_push TOP1 TARGET  -> exit 0 if top1 is a prefix of target (daemon's _is_push)
  python3 -c "import sys; t1=sys.argv[1].strip().lower(); tg=sys.argv[2].strip().lower(); sys.exit(0 if (len(t1)>=1 and tg.startswith(t1)) else 1)" "$1" "$2"
}

# verify_pass NAME -> sets globals: fail (0/1), leaked (0/1). Prints target + held-out tables.
verify_pass() {
  echo "--- verify ($1) ---"
  fail=0; leaked=0
  echo "  targets (want PUSH):"
  for f in "${FACTS[@]}"; do
    ent="${f%%$'\t'*}"; tok="${f##*$'\t'}"
    IFS=$'\t' read -r t1 p1 <<<"$(top1 "$(prompt_of "$ent")")"
    if is_push "$t1" "$tok"; then v=PUSH; else v="MISS($tok)"; fail=1; fi
    printf '    %-10s -> %-12s (%.0f%%)  %s\n' "$ent" "$t1" "$(python3 -c "print($p1*100)")" "$v"
  done
  echo "  held-outs (want NATIVE = no leak to any edited target):"
  for ho in "${HELDOUTS[@]}"; do
    IFS=$'\t' read -r t1 p1 <<<"$(top1 "$(prompt_of "$ho")")"
    leak=""
    for f in "${FACTS[@]}"; do
      tok="${f##*$'\t'}"
      if is_push "$t1" "$tok"; then leak="LEAK->$tok"; fail=1; leaked=1; break; fi
    done
    printf '    %-10s -> %-12s (%.0f%%)  %s\n' "$ho" "$t1" "$(python3 -c "print($p1*100)")" "${leak:-native}"
  done
}

echo "--- 4. verify (post-recalibrate, fast path) ---"
verify_pass post-recalibrate

# /reorthogonalize is a REPAIR for emergent cross-fact leak — expensive (re-gates + re-calibrates the
# whole constellation, ~N×(contrasts+N) forward passes). Run it ONLY when a held-out actually leaks;
# the recalibrate-only path is clean in the common case.
if [ "$leaked" -ne 0 ]; then
  echo "--- 5. leak detected -> /reorthogonalize (kill emergent cross-fact leak) ---"
  req POST /reorthogonalize | python3 -c "import sys,json; d=json.load(sys.stdin); print('    reorthogonalized',d.get('reorthogonalized'), d.get('note','')); [print(f\"      {c['target_token']!r}: τ {c['tau_old']}->{c['tau_new']}\") for c in d.get('changes',[])]"
  verify_pass post-reorthogonalize
else
  echo "--- 5. no leak after recalibrate -> /reorthogonalize not needed ---"
fi

echo "--- 6. reset ---"
req POST /reset | jq -c .

if [ "$fail" -eq 0 ]; then
  echo "VERDICT: PASS — all targets pushed, all held-outs native (constellation clean)."
else
  echo "VERDICT: FAIL — see MISS/LEAK above."; exit 2
fi
