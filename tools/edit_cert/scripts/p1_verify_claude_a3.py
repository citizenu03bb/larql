"""Model-free check of Claude's five A3 solves (docs/phase-b-p1-amendment-a3.md), written by Claude Code.

No forward: it reads the saved arrays, C17 and the block-17 c_proj weights from the safetensors file, and checks the
reference algebra (R1, R4, R7-R10), the caps, the input hashes and the completion index.

  python3 scripts/p1_verify_claude_a3.py [ROOT]

Writes ROOT/solves-claude-a3/verify-claude.json.
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
from scripts.p1_verify_claude_solves import c_proj, rel, sha  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027'
N_TEMPLATES, STEPS = 21, 20


def main() -> int:
    cov = json.loads((ROOT / 'covariance/status.json').read_text())
    tstat = json.loads((ROOT / 'templates/status.json').read_text())
    C = np.load(ROOT / 'covariance/C17.npy').astype(np.float64)
    W, b = c_proj()
    index = json.loads((ROOT / 'solves-claude-a3/completion.json').read_text())
    hashes = dict(covariance=sha(ROOT / 'covariance/C17.npy') == cov['covariance_sha256'],
                  templates=sha(ROOT / 'templates/templates.json') == tstat['templates_sha256'])
    facts = {}
    for fact in REQUESTS:
        d = ROOT / 'solves-claude-a3' / fact
        st = json.loads((d / 'status.json').read_text())
        be = json.loads((d / 'edit/backend.json').read_text())
        a = {k: np.load(d / f'{k}.npy').astype(np.float64)
             for k in ('U', 'V', 'k_mean', 'k_clean', 'context_keys', 'delta', 'target_init', 'cur_output')}
        U, u = a['U'][:, 0], a['V'][:, 0]
        q = np.linalg.solve(C, a['k_mean'])
        dot = float(a['k_clean'] @ u)
        target = a['target_init'] + a['delta']
        t = float(np.linalg.norm(a['target_init']))
        tr = st['trace']
        v1_keys = np.load(ROOT / 'solves-claude' / fact / 'context_keys.npy').astype(np.float64)
        ent = index['facts'].get(fact, {})
        checks = dict(
            index_status_hash=ent.get('status_sha256') == sha(d / 'status.json'),
            index_edit_hash=ent.get('edit_content_sha256') == be['content_sha256'] == st['edit_content_sha256'],
            edit_is_the_factors=bool(np.array_equal(np.load(d / 'edit/U0.npy'), np.load(d / 'U.npy'))
                                     and np.array_equal(np.load(d / 'edit/V0.npy'), np.load(d / 'V.npy'))),
            edit_target=be['params']['targets'] == [dict(layer=17, module='transformer.h.17.mlp.c_proj', scale=1.0)],
            inputs_pinned=st['covariance_sha256'] == cov['covariance_sha256'] and st['templates_sha256'] == tstat['templates_sha256'],
            forwards_exact=st['forwards'] == N_TEMPLATES + STEPS * (N_TEMPLATES + 1) + 1 <= SEQ_CAP,
            steps_exact=st['gradient_steps'] == len(tr) == STEPS <= STEP_CAP,
            k_mean_is_mean_key=rel(a['context_keys'].mean(axis=0), a['k_mean']) < 1e-5,
            keys_match_p13b=rel(a['context_keys'], v1_keys) < 1e-6,
            k_clean_is_template0_key=rel(a['k_clean'], a['context_keys'][0]) < 1e-3,
            cur_output_is_module_image=rel(a['k_clean'] @ W + b, a['cur_output']) < 2e-2,   # module ran in fp16
            R1_unit_C_inverse_k_mean=rel(u, q / np.linalg.norm(q)) < 1e-5 and abs(np.linalg.norm(u) - 1) < 1e-5,
            R9_right_vector=rel((target - a['cur_output']) / dot, U) < 1e-5,
            R9_constraint=rel(a['cur_output'] + U * dot, target) < 1e-5,
            R7_unsquared_decay=tr[0]['decay'] == 0 and all(
                abs(y['decay'] - 0.5 * x['delta_norm'] / t ** 2) <= 1e-3 * max(y['decay'], 1e-12) for x, y in zip(tr, tr[1:])),
            R8_clamp=float(np.linalg.norm(a['delta'])) <= 4 * t * (1 + 1e-4),
            delta_matches_trace=abs(float(np.linalg.norm(a['delta'])) - tr[-1]['delta_norm']) <= 1e-3 * tr[-1]['delta_norm'],
            finite_trace=all(np.isfinite(x['loss']) for x in tr),
        )
        checks = {k: bool(v) for k, v in checks.items()}
        facts[fact] = dict(ok=all(checks.values()), checks=checks,
                           readings=dict(loss_first=tr[0]['loss'], loss_last=tr[-1]['loss'], nll_last=tr[-1]['nll'],
                                         division_factor=dot, norm_delta=float(np.linalg.norm(a['delta'])),
                                         norm_target_init=t, UV=float(np.linalg.norm(U) * np.linalg.norm(u)),
                                         cur_output_rel_error=rel(a['k_clean'] @ W + b, a['cur_output']),
                                         forwards=st['forwards'], seconds=st['seconds']))
    ok = all(hashes.values()) and index.get('complete') is True and all(f['ok'] for f in facts.values())
    out = dict(author='Claude Code (verifier and solves by the same agent; Codex cross-checks at P1-4)', ok=ok,
               index_complete=index.get('complete'), hashes=hashes, facts=facts,
               total_forwards=sum(f['readings']['forwards'] for f in facts.values()))
    (ROOT / 'solves-claude-a3/verify-claude.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(ok=ok, index_complete=index.get('complete'), hashes=hashes,
                          facts={k: [c for c, v in f['checks'].items() if not v] or 'ok' for k, f in facts.items()},
                          total_forwards=out['total_forwards'])))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
