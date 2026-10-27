"""Model-free check of Claude's five P1 solves (docs/phase-b-p1-amendment-a1.md), written by Claude Code.

No forward: it reads the saved factors, the P1-2 covariance and the block-17 c_proj weights from the safetensors
file, and checks the rank-1 algebra, the caps, the input hashes and the completion index.

  python3 scripts/p1_verify_claude_solves.py [ROOT]

Writes ROOT/solves-claude/verify-claude.json. It holds norms and losses, so like <fact>/status.json it is not for
Codex before P1-4; only the pass/fail line goes into completion-facing reports.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from p1_rome_claude.run import REQUESTS, SEQ_CAP, STEP_CAP  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027'
WEIGHTS = Path('/mnt/models/hf-models/gpt2-xl/model.safetensors')
N_TEMPLATES, STEPS = 21, 20


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def c_proj():
    from safetensors import safe_open
    with safe_open(WEIGHTS, 'np') as f:   # only the two tensors, not the whole file
        pre = 'transformer.' if 'transformer.h.17.mlp.c_proj.weight' in f.keys() else ''
        return (f.get_tensor(f'{pre}h.17.mlp.c_proj.weight').astype(np.float64),
                f.get_tensor(f'{pre}h.17.mlp.c_proj.bias').astype(np.float64))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> int:
    cov = json.loads((ROOT / 'covariance/status.json').read_text())
    tstat = json.loads((ROOT / 'templates/status.json').read_text())
    C = np.load(ROOT / 'covariance/C17.npy').astype(np.float64)
    W, b = c_proj()   # GPT-2 Conv1D: y = x @ W + b, W is (6400, 1600)
    index = json.loads((ROOT / 'solves-claude/completion.json').read_text())
    hashes = dict(covariance=sha(ROOT / 'covariance/C17.npy') == cov['covariance_sha256'],
                  templates=sha(ROOT / 'templates/templates.json') == tstat['templates_sha256'])
    facts = {}
    for fact in REQUESTS:
        d = ROOT / 'solves-claude' / fact
        st = json.loads((d / 'status.json').read_text())
        be = json.loads((d / 'edit/backend.json').read_text())
        a = {k: np.load(d / f'{k}.npy').astype(np.float64) for k in ('U', 'V', 'k_star', 'v_star', 'z0', 'context_keys')}
        U, V, k, v, z0 = a['U'][:, 0], a['V'][:, 0], a['k_star'], a['v_star'], a['z0']
        q = np.linalg.solve(C, k)
        V_ref = q / (k @ q)
        ent = index['facts'].get(fact, {})
        checks = dict(
            index_status_hash=ent.get('status_sha256') == sha(d / 'status.json'),
            index_edit_hash=ent.get('edit_content_sha256') == be['content_sha256'] == st['edit_content_sha256'],
            edit_is_the_factors=bool(np.array_equal(np.load(d / 'edit/U0.npy'), np.load(d / 'U.npy'))
                                     and np.array_equal(np.load(d / 'edit/V0.npy'), np.load(d / 'V.npy'))),
            edit_target=be['params']['targets'] == [dict(layer=17, module='transformer.h.17.mlp.c_proj', scale=1.0)],
            inputs_pinned=st['covariance_sha256'] == cov['covariance_sha256'] and st['templates_sha256'] == tstat['templates_sha256'],
            forwards_exact=st['forwards'] == N_TEMPLATES + 1 + STEPS * (N_TEMPLATES + 1) <= SEQ_CAP,
            steps_exact=st['gradient_steps'] == len(st['trace']) == STEPS <= STEP_CAP,
            k_star_is_mean_key=rel(a['context_keys'].mean(axis=0), k) < 1e-5,
            z0_is_module_image=rel(k @ W + b, z0) < 2e-2,              # module ran in fp16
            V_is_C_inverse_k=rel(V, V_ref) < 1e-3,
            unit_dose=abs(V @ k - 1) < 1e-3,
            U_is_v_minus_z0=rel(v - z0, U) < 1e-5,
            constraint=rel(z0 + U * (V @ k), v) < 1e-5,
            clamp=float(np.linalg.norm(v - z0)) <= 4 * float(np.linalg.norm(z0)) * (1 + 1e-4),
            finite_trace=all(np.isfinite(t['loss']) for t in st['trace']),
        )
        facts[fact] = dict(ok=all(checks.values()), checks=checks,
                           readings=dict(nll_first=st['trace'][0]['nll'], nll_last=st['trace'][-1]['nll'],
                                         z0_rel_error=rel(k @ W + b, z0), V_rel_error=rel(V, V_ref),
                                         norm_U=float(np.linalg.norm(U)), norm_V=float(np.linalg.norm(V)),
                                         UV=float(np.linalg.norm(U) * np.linalg.norm(V)),
                                         shift_over_z0=float(np.linalg.norm(v - z0) / np.linalg.norm(z0)),
                                         forwards=st['forwards'], seconds=st['seconds']))
    ok = all(hashes.values()) and index.get('complete') is True and all(f['ok'] for f in facts.values())
    out = dict(author='Claude Code (verifier and solves by the same agent; Codex cross-checks at P1-4)', ok=ok,
               index_complete=index.get('complete'), hashes=hashes, facts=facts,
               total_forwards=sum(f['readings']['forwards'] for f in facts.values()))
    (ROOT / 'solves-claude/verify-claude.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(ok=ok, index_complete=index.get('complete'), hashes=hashes,
                          facts={k: v['ok'] for k, v in facts.items()}, total_forwards=out['total_forwards'])))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
