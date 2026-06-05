#!/usr/bin/env bash
# End-to-end smoke of /reorthogonalize: fixes the EMERGENT constellation leak (a held-out leaking to an
# edited target because slots interact, though neither leaks alone). Installs Atlantis→Poseidon +
# Krypton→Superman (Egypt→Superman leaks), calls /reorthogonalize (mutual cross-fact re-gate + τ), and
# checks the leak is gone while both targets still push.
set -euo pipefail
H=http://127.0.0.1:8770
jq_top() { python3 -c "import sys,json; d=json.load(sys.stdin); p=d['predictions'][0]; print(f\"{p['token']!r} {p['prob']:.0%}\")"; }
BASE='"The capital of Olympus is","The capital of France is","The capital of Japan is","The capital of Italy is","The capital of Brazil is"'
infer_all() { for P in Atlantis Krypton Egypt Spain Germany; do
  echo -n "    $P : "; curl -s -X POST $H/infer -d "{\"prompt\":\"The capital of $P is\"}" | jq_top; done; }

curl -s -X POST $H/reset >/dev/null
echo "=== install Atlantis→Poseidon + Krypton→Superman (base contrasts) ==="
curl -s -X POST $H/edit -d "{\"target_prompt\":\"The capital of Atlantis is\",\"target_token\":\"Poseidon\",\"contrast_prompts\":[$BASE]}" >/dev/null
curl -s -X POST $H/edit -d "{\"target_prompt\":\"The capital of Krypton is\",\"target_token\":\"Superman\",\"contrast_prompts\":[$BASE]}" >/dev/null
echo "  BEFORE /reorthogonalize (expect Egypt→Superman LEAK):"; infer_all

echo "=== POST /reorthogonalize ==="
curl -s -X POST $H/reorthogonalize | python3 -c "import sys,json; d=json.load(sys.stdin); print('  reorthogonalized',d['reorthogonalized']); [print(f\"    {c['target_token']!r}: τ {c['tau_old']}->{c['tau_new']}\") for c in d.get('changes',[])]"

echo "  AFTER /reorthogonalize (expect NO held-out → Superman/Poseidon; both targets push):"; infer_all
curl -s -X POST $H/reset >/dev/null; echo "(reset)"
echo "VERDICT: PASS if Atlantis→Pose & Krypton→Superman push AND Egypt/Spain/Germany do NOT say Superman/Poseidon"
