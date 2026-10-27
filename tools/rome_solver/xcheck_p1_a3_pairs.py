#!/usr/bin/env python3
"""Independent model-free A3 comparison; frozen P1-4 math is unchanged.

New I/O for the A3 witness layout. No peer comparator/solver/verifier code
or comparison reports are read. No model execution or candidate mutation.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from p1_pair_math import affine_constraint, compare_factors
from xcheck_p1_pairs import ROOT, FACTS, C_SHA, T_SHA, MODEL_SHA, array, edit, read, require, run as original_audit, sha

CONVENTIONS_COMMIT = '60482963'
CONVENTIONS_PATH = 'docs/phase-b-p1-a3-conventions-codex.md'
CONVENTIONS_RECEIPT = 'docs/research-data/phase-b-p1-a3-conventions-codex.json'
COMPLETION_SHA = '725a06742a5c22e644ee980ebd560516fbc7f4f9c7d9e5d0ef8405845f7a504d'
BENCH_CAPS = dict(zip(FACTS, (832, 784, 784, 832, 832)))


def run(root, repo):
    # Verify the declaration really exists in its committed pre-comparison form.
    committed = subprocess.check_output(['git', 'show', CONVENTIONS_COMMIT + ':' + CONVENTIONS_PATH], cwd=repo)
    require((repo / CONVENTIONS_PATH).read_bytes() == committed, 'Conventions changed since freeze')
    receipt = read(repo / CONVENTIONS_RECEIPT)
    require(sha(repo / CONVENTIONS_PATH) == receipt['document_sha256'], 'Conventions receipt mismatch')
    for path, expected in receipt['source_sha256'].items():
        require(sha(repo / path) == expected, 'Executed Codex source changed')
    frozen_original = read(repo / 'docs/research-data/phase-b-p1-p4-pair-xcheck-codex.json')
    require(original_audit(root) == frozen_original, 'Original ten candidates or gate changed')
    peer_root = root / 'solves-claude-a3'
    completion_path = peer_root / 'completion.json'
    require(sha(completion_path) == COMPLETION_SHA, 'A3 completion changed')
    require(completion_path.read_bytes() == (root / 'solves-claude-a3-peer-completion-codex-before.json').read_bytes(), 'A3 completion snapshot mismatch')
    completion = read(completion_path)
    require(completion['complete'] is True and set(completion['facts']) == set(FACTS), 'A3 completion gate closed')
    result = dict(author='Codex', stage='P1-4 A3', model_forwards=0,
        conventions_commit=CONVENTIONS_COMMIT, conventions_document_sha256=receipt['document_sha256'],
        math_frozen_commit='c713a033', peer_source_or_comparison_read=False,
        peer_completion_sha256=COMPLETION_SHA, original_p1_4_unchanged=True,
        covariance_sha256=C_SHA, templates_sha256=T_SHA, model_sha256=MODEL_SHA,
        primary_peer_target_convention='float64(target_init) + float64(delta), following declared float64 U construction; native float32 sum reported diagnostically, never used to rescore agreement',
        cases={})
    for fact in FACTS:
        own = root / 'solves-codex' / fact; other = peer_root / fact
        seal = completion['facts'][fact]
        require(seal['state'] == 'complete' and seal['forwards'] == 462 and seal['gradient_steps'] == 20, 'A3 completion counters')
        require(sha(other / 'status.json') == seal['status_sha256'], 'A3 status hash mismatch')
        s = read(other / 'status.json'); m = read(own / 'solver.json')
        require(s['fact'] == fact and s['forwards'] == 462 and s['gradient_steps'] == 20, 'A3 status/counters')
        require(s['reference'] == 'kmeng01/rome@0874014c' and s['exception'] == '20 updates, no early stop (A3 §3)', 'A3 reference/policy mismatch')
        require(s['covariance_sha256'] == C_SHA and s['templates_sha256'] == T_SHA, 'A3 input identity')
        require(s['hp'] == dict(layer=17, steps=20, lr=.5, decay=.5, clamp=4., kl=.0625, n_short=10, len_short=5, n_long=10, len_long=10), 'A3 hyperparameters changed')
        fingerprint = dict(line.split('  ', 1)[::-1] for line in s['fingerprint']['weights_sha256'].splitlines())
        require(fingerprint['model.safetensors'] == MODEL_SHA and s['fingerprint']['dtype'] == 'float16', 'A3 model identity')
        ob, oa = edit(own); pb, pa = edit(other / 'edit')
        require(pb['content_sha256'] == s['edit_content_sha256'] == seal['edit_content_sha256'], 'A3 edit identity')
        for name, backend_name in [('U', 'U0'), ('V', 'V0')]:
            require(np.array_equal(array(other / (name + '.npy')), pa[backend_name]), 'A3 witness differs from served factor')
        k = array(other / 'k_clean.npy'); keys = array(other / 'context_keys.npy')
        own_keys = array(own / 'context_keys.npy')
        require(keys.shape == (21, 6400) and k.shape == (6400,), 'A3 key dimensions')
        require(np.array_equal(k, keys[0]), 'A3 constraint is not the clean key')
        require(s['positions'] == m['positions'] + [m['essence_position']], 'A3 subject positions differ')
        initial = array(other / 'target_init.npy'); delta = array(other / 'delta.npy'); base = array(other / 'cur_output.npy')
        target64 = initial.astype(np.float64) + delta.astype(np.float64)
        target32 = initial + delta
        peer_algebra = affine_constraint(pa['U0'], pa['V0'], k, target64, base)
        own_algebra = affine_constraint(oa['U0'], oa['V0'], array(own / 'k_star.npy'), array(own / 'v_star.npy'), array(own / 'w_k_star.npy'))
        comparison = compare_factors(oa['U0'], oa['V0'], pa['U0'], pa['V0'])
        result['cases'][fact] = dict(comparison=comparison,
            algebra_codex=own_algebra, algebra_claude_a3=peer_algebra,
            algebra_claude_float32_target_diagnostic=affine_constraint(pa['U0'], pa['V0'], k, target32, base),
            old_own_manifest_sha256=sha(own / 'solver.json'), peer_status_sha256=sha(other / 'status.json'),
            own_content_sha256=ob['content_sha256'], peer_content_sha256=pb['content_sha256'],
            peer_observed_file_sha256={str(p.relative_to(other)): sha(p) for p in sorted(other.rglob('*')) if p.is_file()},
            peer_witness_provenance='Completion pins status and edit content; remaining witness file hashes recorded on first independent comparison.',
            context_keys_exactly_equal=bool(np.array_equal(own_keys, keys)),
            peer_initial_minus_singleton_base_l2=float(np.linalg.norm(initial.astype(np.float64) - base.astype(np.float64))),
            peer_base_minus_codex_base_max_abs=float(np.max(np.abs(base.astype(np.float64) - array(own / 'w_k_star.npy').astype(np.float64)))),
            stored_V_clean_dot=float(pa['V0'].astype(np.float64)[:, 0] @ k.astype(np.float64)),
            reported_division_factor=s['division_factor'],
            peer_reported_constraint_error=s['constraint_error'],
            same_subject_positions=True, raw_sequence_forwards=s['forwards'], gradient_steps=s['gradient_steps'],
            p1_5_bench_cap=BENCH_CAPS[fact], p1_5_probe_cap=54)
    result['agreement_count'] = sum(c['comparison']['agrees'] is True for c in result['cases'].values())
    result['disagreement_count'] = sum(c['comparison']['agrees'] is False for c in result['cases'].values())
    result['ambiguous_count'] = 5 - result['agreement_count'] - result['disagreement_count']
    result['all_affine_constraints_pass'] = all(c['algebra_codex']['passed'] and c['algebra_claude_a3']['passed'] for c in result['cases'].values())
    result['p1_closes_no_A4'] = result['disagreement_count'] >= 3
    result['p1_5_eligible_facts'] = [] if result['p1_closes_no_A4'] or not result['all_affine_constraints_pass'] or result['ambiguous_count'] else [f for f, c in result['cases'].items() if c['comparison']['agrees'] is True]
    result['p1_5_total_cap_for_eligible_facts'] = sum(BENCH_CAPS[f] + 54 for f in result['p1_5_eligible_facts'])
    result['forwards_spent_since_v2'] = 6720 + sum(f['forwards'] for f in completion['facts'].values())
    result['interpretation'] = 'Agreement is a replication gate for candidate construction, not certification. No candidate or threshold changed, no new solve, no rescore under altered conventions. Explanatory conjectures must use the declarations frozen before this comparison.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.root, Path(__file__).resolve().parents[2])
    # Preserve a first run rather than overwriting its scores.
    with args.out.open('x') as f:
        f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('agreement_count', 'disagreement_count', 'ambiguous_count', 'all_affine_constraints_pass', 'p1_closes_no_A4', 'p1_5_eligible_facts', 'p1_5_total_cap_for_eligible_facts')}))


if __name__ == '__main__':
    main()
