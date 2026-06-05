#!/usr/bin/env bash
# End-to-end smoke test of the /recalibrate endpoint (added 2026-06-03, never smoke-tested).
# Installs a 2-fact constellation, calls /recalibrate (co-active push-τ over all slots in the dense
# regime), and checks: it returns trails without error, both targets still push, held-outs stay clean.
set -euo pipefail
H=http://127.0.0.1:8770   # 8765 = Aftertone TTS daemon — LARQL on 8770
jq_top() { python3 -c "import sys,json; d=json.load(sys.stdin); p=d['predictions'][0]; print(f\"{p['token']!r} {p['prob']:.0%}\")"; }
edit_summary() { python3 -c "import sys,json; d=json.load(sys.stdin); r=d['record']; print('  tau',r['tau'],'calibrated',r['tau_calibrated'],'layers',r['layers'][:2],'...')"; }

echo "=== install fact 1: Atlantis→Poseidon ==="
curl -s -X POST $H/edit -d '{
  "target_prompt":"The capital of Atlantis is","target_token":"Poseidon",
  "contrast_prompts":["The capital of Olympus is","The capital of France is","The capital of Japan is","The capital of Italy is","The capital of Brazil is"]
}' | edit_summary

echo "=== install fact 2: Krypton→Superman ==="
curl -s -X POST $H/edit -d '{
  "target_prompt":"The capital of Krypton is","target_token":"Superman",
  "contrast_prompts":["The capital of Olympus is","The capital of France is","The capital of Japan is","The capital of Italy is","The capital of Brazil is"]
}' | edit_summary

echo "=== pre-recalibrate /infer ==="
echo -n "  Atlantis(→Poseidon): "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Atlantis is"}' | jq_top
echo -n "  Krypton(→Superman) : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Krypton is"}' | jq_top
echo -n "  Spain (held-out)   : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Spain is"}' | jq_top
echo -n "  Egypt (held-out)   : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Egypt is"}' | jq_top

echo "=== POST /recalibrate (the untested endpoint) ==="
curl -s -X POST $H/recalibrate | python3 -c "import sys,json; d=json.load(sys.stdin); print('  recalibrated',d['recalibrated']); [print(f\"    {c['target_token']!r}: τ {c['tau_old']} -> {c['tau_new']}  changed={c['changed']}\") for c in d['changes']]"

echo "=== post-recalibrate /infer (targets still push, held-outs still clean) ==="
echo -n "  Atlantis(→Poseidon): "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Atlantis is"}' | jq_top
echo -n "  Krypton(→Superman) : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Krypton is"}' | jq_top
echo -n "  Spain (held-out)   : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Spain is"}' | jq_top
echo -n "  Egypt (held-out)   : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Egypt is"}' | jq_top

echo "=== /reset ==="
curl -s -X POST $H/reset; echo
echo "VERDICT: recalibrate ran end-to-end (inspect τ changes + that pushes/held-outs survived)"
