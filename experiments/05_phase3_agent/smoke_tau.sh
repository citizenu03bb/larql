#!/usr/bin/env bash
# End-to-end smoke test of the τ-calibrated /edit endpoint.
# Edits Atlantis→Poseidon (the leak config), then checks: target pushes, held-outs clean.
set -euo pipefail
H=http://127.0.0.1:8770   # 8765 is taken by the Aftertone TTS daemon — use 8770 for LARQL
jq_top() { python3 -c "import sys,json; d=json.load(sys.stdin); p=d['predictions'][0]; print(f\"{p['token']!r} {p['prob']:.0%}\")"; }

echo "--- /edit Atlantis→Poseidon (auto-calibrate τ) ---"
curl -s -X POST $H/edit -d '{
  "target_prompt":"The capital of Atlantis is","target_token":"Poseidon",
  "contrast_prompts":["The capital of Olympus is","The capital of France is","The capital of Japan is","The capital of Italy is","The capital of Brazil is"]
}' | python3 -c "import sys,json; d=json.load(sys.stdin); r=d['record']; print('  layers',r['layers']); print('  tau',r['tau'],'calibrated',r['tau_calibrated']); print('  trail',[(t['tau'],t['top1'].strip(),round(t['prob'],2),t['push']) for t in (d.get('tau_trail') or [])])"

echo "--- /infer target (expect Poseidon push) ---"
echo -n "  Atlantis : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Atlantis is"}' | jq_top
echo "--- /infer held-outs (expect clean / native, NO Poseidon) ---"
echo -n "  Belgium  : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Belgium is"}' | jq_top
echo -n "  Spain    : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Spain is"}' | jq_top
echo -n "  Egypt    : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Egypt is"}' | jq_top

echo "--- /reset ---"
curl -s -X POST $H/reset
echo
echo -n "  post-reset Atlantis : "; curl -s -X POST $H/infer -d '{"prompt":"The capital of Atlantis is"}' | jq_top
