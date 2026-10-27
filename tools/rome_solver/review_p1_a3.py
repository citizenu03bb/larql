#!/usr/bin/env python3
"""Model-free review of brief 42: factor decomposition and P1 budget.

Reads archived candidates, not peer implementation code. Does not amend or
re-run either solve. The proposed official reference was read as text only.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from p1_pair_math import cosine
from xcheck_p1_pairs import ROOT, array, read, require, run, sha

PEER_REVISION = '7d9d3b866f5ee8fdc07a5b9cc4e55401b21c4aa8'
REFERENCE_REVISION = '0874014cd9837e4365f3e6f3c71400ef11509e04'
PEER_REPORT = 'docs/research-data/phase-b-p1-p4-xcheck-claude.json'


def factors(delta, peer_shift, unit_v, clean, mean, norm_ratio):
    unit_v = np.asarray(unit_v, dtype=np.float64).ravel()
    unit_v = unit_v / np.linalg.norm(unit_v)
    clean_dot = float(unit_v @ clean.astype(np.float64))
    mean_dot = float(unit_v @ mean.astype(np.float64))
    require(clean_dot != 0 and mean_dot != 0, 'Degenerate constraint')
    objective = float(np.linalg.norm(delta.astype(np.float64)) / np.linalg.norm(peer_shift.astype(np.float64)))
    key = abs(mean_dot / clean_dot)
    product = objective * key
    return dict(objective_factor=objective, key_factor=key, product=product,
                u_dot_k_mean=mean_dot, u_dot_k_clean=clean_dot,
                measured_codex_over_claude_norm=norm_ratio,
                product_relative_error=abs(product / norm_ratio - 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    original = read(repo / 'docs/research-data/phase-b-p1-p4-pair-xcheck-codex.json')
    current = run(ROOT)  # Recheck completion gates and all previously declared hashes.
    require(current == original, 'The frozen P1-4 replay changed')
    peer_bytes = subprocess.check_output(['git', 'show', f'{PEER_REVISION}:{PEER_REPORT}'], cwd=repo)
    peer_report = json.loads(peer_bytes)
    output = dict(author='Codex', model_forwards=0, amendment_approved=False,
                  frozen_p1_4_unchanged=True, peer_source_read=False,
                  peer_report_revision=PEER_REVISION, peer_report_path=PEER_REPORT,
                  reference_revision_read_as_text=REFERENCE_REVISION, facts={})
    for fact, previous in current['cases'].items():
        own = ROOT / 'solves-codex' / fact; other = ROOT / 'solves-claude' / fact
        # The peer's witnesses were first hashed at P1-4, and are rechecked here.
        for name, expected in original['cases'][fact]['peer_observed_file_sha256'].items():
            require(sha(other / name) == expected, 'Peer artifact changed since P1-4')
        delta = array(own / 'delta.npy')
        peer_shift = array(other / 'v_star.npy').astype(np.float64) - array(other / 'z0.npy').astype(np.float64)
        unit_v = array(own / 'V0.npy')
        clean = array(own / 'k_star.npy'); mean = array(own / 'context_keys.npy').mean(axis=0, dtype=np.float64)
        comparison = previous['comparison']
        norm_ratio = comparison['norm_a'] / comparison['norm_b']
        d = factors(delta, peer_shift, unit_v, clean, mean, norm_ratio)
        d.update(cos_delta_U_codex=cosine(delta, array(own / 'U0.npy')),
                 ratio=norm_ratio, clean_subject_last_position=read(own / 'solver.json')['positions'][0])
        names = ('objective_factor', 'key_factor', 'product', 'u_dot_k_mean', 'u_dot_k_clean', 'cos_delta_U_codex', 'ratio')
        d['comparison_to_peer_report'] = {
            name: dict(codex=d[name], claude=peer_report['facts'][fact][name],
                       absolute_difference=abs(d[name] - peer_report['facts'][fact][name]),
                       equal_exactly=d[name] == peer_report['facts'][fact][name],
                       within_float32_tolerance=bool(np.isclose(d[name], peer_report['facts'][fact][name], rtol=2e-5, atol=2e-5)))
            for name in names}
        output['facts'][fact] = d
    qual = read(repo / 'docs/research-data/phase-b-p1-v2-qualification-xcheck-codex.json')['forwards']
    templates = read(ROOT / 'templates/status.json')['forwards']
    cov = read(ROOT / 'covariance/status.json')['total_sequence_forwards']
    own = read(ROOT / 'solves-codex-index.json')['model_sequence_forwards']
    peer = sum(f['forwards'] for f in read(ROOT / 'solves-claude/completion.json')['facts'].values())
    current_v2 = qual + templates + cov + own + peer
    output['budget'] = dict(qualification_v2=qual, templates=templates, covariance_all_attempts=cov,
        solves_codex=own, solves_claude=peer, current_since_v2=current_v2,
        qualification_v1_reported=500, qualification_v1_source=f'{PEER_REVISION}:docs/phase-b-p1-qualification-results.md',
        current_entire_p1_including_reported_v1=current_v2 + 500,
        proposed_new_solves_expected=5 * 462, proposed_new_solves_service_cap_total=5 * 500,
        p1_5_estimate_not_a_frozen_cap=3000,
        projected_since_v2=current_v2 + 2310 + 3000,
        projected_entire_p1_including_reported_v1=current_v2 + 500 + 2310 + 3000)
    output['all_peer_numbers_within_tolerance'] = all(v['within_float32_tolerance'] for f in output['facts'].values() for v in f['comparison_to_peer_report'].values())
    output['max_decomposition_relative_error'] = max(f['product_relative_error'] for f in output['facts'].values())
    output['review'] = dict(support_a3_in_principle=True, required_clarifications=[
        'Keep A1: 20 actual Adam updates, no early stopping. Explicit exception to the official loop (at most 19 updates with v_num_grad_steps=20). Literal official-loop behaviour cannot be silently compared to the frozen Codex candidates.',
        'Pin an immutable official revision and retain explicit common covariance/templates, hook, KL, clean affine constraint and numerical conventions. The pinned code is a mathematical reference, not a claim of bitwise reproduction.',
        'Recount the ledger including the discarded 272 covariance forwards; disclose the v1 qualification separately. Freeze a new total cap and P1-5 allocation rather than treating its 3000-forward estimate as a cap.'
    ], causal_limit='The norm product is an algebraic decomposition, within serialization error; it does not isolate which objective change caused the optimization difference. Position zero is observed, not an ablated causal explanation.')
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({key: output[key] for key in ('all_peer_numbers_within_tolerance', 'max_decomposition_relative_error', 'budget')}))


if __name__ == '__main__':
    main()
