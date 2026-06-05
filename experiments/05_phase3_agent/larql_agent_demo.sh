#!/usr/bin/env bash
# larql_agent_demo.sh — FAST single-fact self-editing demo for an autonomous agent to drive
# in one bash call (no slow reorthogonalize). Held-outs are placed IN the contrast set, so the
# QR-projection gate is structurally orthogonal to them (0% off-diagonal leak by construction) —
# the edit is clean without the expensive repair pass. ~3-5 min.
#
#   reset → /edit Atlantis→Poseidon (contrasts incl. the held-outs) → verify target push
#         → verify held-outs native → reset → VERDICT
#
# Port: $LARQL_PORT (default 8770). Needs the daemon running.
set -euo pipefail
H="http://127.0.0.1:${LARQL_PORT:-8770}"

TARGET_PROMPT="The capital of Atlantis is"
TARGET_TOKEN="Poseidon"
# contrast set INCLUDES the held-outs we verify, so they are structurally protected
CONTRASTS='"The capital of Olympus is","The capital of France is","The capital of Japan is","The capital of Egypt is","The capital of Belgium is"'
HELDOUTS=(Egypt Belgium)

top1() { curl -sS --max-time 200 -X POST "$H/infer" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg p "$1" '{prompt:$p, top_k:5}')" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); p=d['predictions'][0]; print(p['token']+'\t'+str(p['prob']))"; }
is_push() { python3 -c "import sys; t1=sys.argv[1].strip().lower(); tg=sys.argv[2].strip().lower(); sys.exit(0 if (len(t1)>=1 and tg.startswith(t1)) else 1)" "$1" "$2"; }

echo "=== single-fact self-editing demo @ $H ==="
curl -sS --max-time 30 -X POST "$H/reset" >/dev/null; echo "reset done"

echo "--- baseline ---"
IFS=$'\t' read -r b1 bp <<<"$(top1 "$TARGET_PROMPT")"
printf '    %-28s -> %s (%.0f%%)\n' "$TARGET_PROMPT" "$b1" "$(python3 -c "print($bp*100)")"

echo "--- edit weights: Atlantis -> Poseidon ---"
tau=$(curl -sS --max-time 400 -X POST "$H/edit" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg p "$TARGET_PROMPT" --arg t "$TARGET_TOKEN" --argjson c "[$CONTRASTS]" \
        '{target_prompt:$p, target_token:$t, contrast_prompts:$c}')" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['record']['tau'])")
echo "    calibrated τ=$tau"

fail=0
echo "--- verify target (want PUSH) ---"
IFS=$'\t' read -r t1 tp <<<"$(top1 "$TARGET_PROMPT")"
if is_push "$t1" "$TARGET_TOKEN"; then v=PUSH; else v="MISS($TARGET_TOKEN)"; fail=1; fi
printf '    Atlantis -> %-12s (%.0f%%)  %s\n' "$t1" "$(python3 -c "print($tp*100)")" "$v"

echo "--- verify held-outs (want NATIVE) ---"
for ho in "${HELDOUTS[@]}"; do
  IFS=$'\t' read -r h1 hp <<<"$(top1 "The capital of $ho is")"
  if is_push "$h1" "$TARGET_TOKEN"; then leak="LEAK->$TARGET_TOKEN"; fail=1; else leak="native"; fi
  printf '    %-8s -> %-12s (%.0f%%)  %s\n' "$ho" "$h1" "$(python3 -c "print($hp*100)")" "$leak"
done

curl -sS --max-time 30 -X POST "$H/reset" >/dev/null; echo "reset done"
if [ "$fail" -eq 0 ]; then
  echo "VERDICT: PASS — Atlantis pushed Poseidon, held-outs stayed native (selective weight edit)."
else
  echo "VERDICT: FAIL — see MISS/LEAK above."; exit 2
fi
