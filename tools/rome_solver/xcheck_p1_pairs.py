#!/usr/bin/env python3
"""Model-free P1-4 artifact audit, after both five-solve completion gates.

No peer source, solver, harness, torch or model execution is imported.
Threshold math was committed at c713a033 before peer factor access.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from p1_pair_math import affine_constraint, compare_factors

ROOT = Path('/home/pat/Projects/larql-research/phase-b-p1-v2-20261027')
FACTS = ('p1-bmw-origin-italy', 'p1-germany-language-finnish',
         'p1-gretzky-sport-baseball', 'p1-nintendo-origin-france',
         'p1-shakespeare-nationality-spanish')
MODEL_SHA = '0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a'
C_SHA = 'a9c6faa06c9b9155e764a9a38298c62c0a11379369bb3869075a76f61057c715'
T_SHA = 'b7dce8e1fa21e362e63e4ea45fc5a25030cce7a61781dcdd55f0c0216afebf15'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array(path):
    a = np.load(path, allow_pickle=False)
    require(bool(np.isfinite(a).all()), f'Nonfinite array: {path}')
    return a


def edit(path):
    """Recalculate the documented format hash independently of Backend.load."""
    b = read(path / 'backend.json')
    require(b['kind'] == 'low_rank' and b['arrays'] == ['U0', 'V0'], 'Unexpected backend')
    require(b['params']['targets'] == [dict(layer=17, module='transformer.h.17.mlp.c_proj', scale=1.)], 'Unexpected target')
    arrays = {name: array(path / (name + '.npy')) for name in b['arrays']}
    require(arrays['U0'].shape == (1600, 1) and arrays['V0'].shape == (6400, 1), 'Wrong orientation/rank')
    require(all(a.dtype == np.float32 for a in arrays.values()), 'Wrong backend dtype')
    h = hashlib.sha256(json.dumps(dict(kind=b['kind'], params=b['params']), sort_keys=True).encode())
    for name, a in sorted(arrays.items()):
        a = np.ascontiguousarray(a)
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    require(h.hexdigest() == b['content_sha256'], f'Content hash mismatch: {path}')
    return b, arrays


def run(root):
    # No peer artifact access until all own five complete statuses exist.
    idx = read(root / 'solves-codex-index.json')
    require(idx['complete'] is True and [c['fact_id'] for c in idx['cases']] == list(FACTS), 'Own completion gate closed')
    for fact in FACTS:
        require(read(root / 'solves-codex' / fact / 'status.json')['status'] == 'complete', 'Own status incomplete')
    peer_path = root / 'solves-claude/completion.json'
    require(peer_path.read_bytes() == (root / 'solves-codex-peer-completion-before.json').read_bytes(), 'Peer completion changed')
    peer = read(peer_path)
    require(peer['complete'] is True and set(peer['facts']) == set(FACTS), 'Peer completion gate closed')
    require(sha(root / 'covariance/C17.npy') == C_SHA, 'Covariance changed')
    require(sha(root / 'templates/templates.json') == T_SHA, 'Templates changed')
    require(read(root / 'covariance/status.json')['status'] == 'complete', 'Incomplete covariance')
    result = dict(author='Codex', stage='P1-4', model_forwards=0,
                  math_frozen_commit='c713a033', peer_source_read=False,
                  peer_completion_sha256=sha(peer_path),
                  own_index_sha256=sha(root / 'solves-codex-index.json'),
                  covariance_sha256=C_SHA, templates_sha256=T_SHA, model_sha256=MODEL_SHA,
                  own_sequence_forwards=idx['model_sequence_forwards'], own_gradient_steps=idx['gradient_steps'], cases={})
    for entry in idx['cases']:
        fact = entry['fact_id']; own = root / 'solves-codex' / fact; other = root / 'solves-claude' / fact
        s = read(own / 'status.json'); m = read(own / 'solver.json'); completion = peer['facts'][fact]
        require(s == entry, 'Own index/status mismatch')
        require(sha(own / 'solver.json') == s['edit_manifest_sha256'], 'Own manifest changed')
        for name, expected in m['artifact_sha256'].items():
            require(sha(own / name) == expected, f'Own payload changed: {fact}/{name}')
        require(sha(other / 'status.json') == completion['status_sha256'], 'Peer status changed')
        p = read(other / 'status.json')
        for status in (s, p):
            require(status['forwards'] == 462 and status['gradient_steps'] == 20, 'Wrong counters')
            require(status['covariance_sha256'] == C_SHA and status['templates_sha256'] == T_SHA, 'Input mismatch')
        require(completion['state'] == 'complete' and completion['forwards'] == 462 and completion['gradient_steps'] == 20, 'Peer completion counters')
        require(m['model_sha256'] == MODEL_SHA, 'Own model mismatch')
        fingerprint = dict(line.split('  ', 1)[::-1] for line in p['fingerprint']['weights_sha256'].splitlines())
        require(fingerprint['model.safetensors'] == MODEL_SHA and p['fingerprint']['dtype'] == 'float16', 'Peer model mismatch')
        ob, oa = edit(own); pb, pa = edit(other / 'edit')
        require(ob['content_sha256'] == s['content_sha256'] == m['content_sha256'], 'Own content identity')
        require(pb['content_sha256'] == p['edit_content_sha256'] == completion['edit_content_sha256'], 'Peer content identity')
        for name, backend_name in [('U', 'U0'), ('V', 'V0')]:
            require(np.array_equal(array(other / (name + '.npy')), pa[backend_name]), 'Peer witness differs from served factor')
        ok = array(own / 'k_star.npy'); pk = array(other / 'k_star.npy')
        ot = array(own / 'v_star.npy'); pt = array(other / 'v_star.npy')
        base = array(own / 'w_k_star.npy'); pbase = array(other / 'z0.npy')
        keys = array(own / 'context_keys.npy'); peer_keys = array(other / 'context_keys.npy')
        require(keys.shape == peer_keys.shape == (21, 6400), 'Context dimensions')
        algebra_own = affine_constraint(oa['U0'], oa['V0'], ok, ot, base)
        algebra_peer = affine_constraint(pa['U0'], pa['V0'], pk, pt, pbase)
        comparison = compare_factors(oa['U0'], oa['V0'], pa['U0'], pa['V0'])
        result['cases'][fact] = dict(comparison=comparison, algebra_codex=algebra_own, algebra_claude=algebra_peer,
            own_manifest_sha256=sha(own / 'solver.json'), own_status_sha256=sha(own / 'status.json'),
            peer_status_sha256=sha(other / 'status.json'), own_content_sha256=ob['content_sha256'], peer_content_sha256=pb['content_sha256'],
            own_payload_hashes_verified=len(m['artifact_sha256']),
            peer_observed_file_sha256={str(p.relative_to(other)): sha(p) for p in sorted(other.rglob('*')) if p.is_file()},
            peer_witness_provenance='Witness npy hashes recorded now; completion previously pins status and edit content only.',
            context_keys_exactly_equal=bool(np.array_equal(keys, peer_keys)),
            own_key_is_first_context=bool(np.array_equal(ok, keys[0])),
            peer_key_minus_mean_l2=float(np.linalg.norm(pk.astype(np.float64) - keys.mean(axis=0, dtype=np.float64))),
            peer_key_minus_clean_l2=float(np.linalg.norm(pk.astype(np.float64) - keys[0])),
            own_clean_dot_V=float(ok.astype(np.float64) @ oa['V0'].astype(np.float64)[:, 0]),
            own_mean_dot_V=float(keys.mean(axis=0, dtype=np.float64) @ oa['V0'].astype(np.float64)[:, 0]),
            peer_decisions=p['decisions'], own_penalty=m['penalty'], own_constraint_key=m['constraint_key'])
    result['agreement_count'] = sum(c['comparison']['agrees'] is True for c in result['cases'].values())
    result['disagreement_count'] = sum(c['comparison']['agrees'] is False for c in result['cases'].values())
    result['ambiguous_count'] = 5 - result['agreement_count'] - result['disagreement_count']
    result['all_algebra_pass'] = all(c['algebra_codex']['passed'] and c['algebra_claude']['passed'] for c in result['cases'].values())
    result['p1_5_blocked_by_majority_disagreement'] = result['disagreement_count'] > 2
    result['interpretation'] = 'No result repaired or re-solved. Shared keys agree, but output factors and update norms fail the predeclared comparison for all five facts. Different objectives/constraint witnesses are visible in saved metadata; causal contributions have not been isolated.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.root)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('agreement_count', 'disagreement_count', 'all_algebra_pass', 'p1_5_blocked_by_majority_disagreement')}))


if __name__ == '__main__':
    main()
