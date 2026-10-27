#!/usr/bin/env python3
"""Audit A3 R1-R11 against frozen Codex sources, witnesses and traces.

No pretrained weights loaded, no model forward, no peer implementation read.
Local tokenization and CPU tensor arithmetic only. Upstream was read as text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from xcheck_p1_pairs import ROOT, array, read, require, run, sha

A3_REV = '21f9ce36'
A3_PATH = 'docs/phase-b-p1-amendment-a3.md'
REF_REV = '0874014cd9837e4365f3e6f3c71400ef11509e04'


def trace_totals(history):
    """All three archived terms already include their coefficients."""
    return [float(h['nll'] + h['kl'] + h['decay']) for h in history]


def subject_last_reference_rule(tok, template, subject):
    """Independent transcription of the prefix/word length rule, not an import."""
    require(template.count('{}') == 1, 'Expected one subject placeholder')
    prefix = template.split('{}')[0]
    word = subject
    if prefix:
        require(prefix.endswith(' '), 'Reference requires a space before the subject')
        prefix = prefix[:-1]
        word = ' ' + subject.strip()
    return len(tok.encode(prefix, add_special_tokens=False)) + len(tok.encode(word, add_special_tokens=False)) - 1


def rounding_example():
    import torch
    base = torch.tensor([1.], dtype=torch.float16)
    delta = torch.tensor([0.0004884], dtype=torch.float32)
    reference = base.clone()
    reference += delta
    codex = base + delta.to(base.dtype)
    return dict(base=base.item(), delta_float32=delta.item(),
                mixed_dtype_inplace_add=reference.item(), cast_delta_before_add=codex.item(),
                equal_exactly=bool(torch.equal(reference, codex)),
                scope='CPU scalar arithmetic example only; no measured effect on the five real solves')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    frozen = read(repo / 'docs/research-data/phase-b-p1-p4-pair-xcheck-codex.json')
    require(run(ROOT) == frozen, 'Frozen P1-4 audit changed')
    rev = subprocess.check_output(['git', 'rev-parse', A3_REV], cwd=repo, text=True).strip()
    spec = subprocess.check_output(['git', 'show', rev + ':' + A3_PATH], cwd=repo)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/mnt/models/hf-models/gpt2-xl', local_files_only=True)
    c = array(ROOT / 'covariance/C17.npy').astype(np.float64)
    result = dict(author='Codex', model_forwards=0, pretrained_weights_loaded=False,
                  peer_implementation_read=False, amendment_approved=False,
                  a3_revision=rev, a3_document_sha256=hashlib.sha256(spec).hexdigest(),
                  reference_revision_read_as_text=REF_REV,
                  frozen_p1_4_replayed_unchanged=True, facts={}, rule_review={})
    for fact in frozen['cases']:
        root = ROOT / 'solves-codex' / fact
        m = read(root / 'solver.json')
        for name, expected in m['source_sha256'].items():
            require(sha(repo / name) == expected, 'Executed Codex source changed: ' + name)
        request = m['request']; templates = m['templates']
        require(len(templates) == 21 and templates[0] == '{}', 'Wrong shared template layout')
        target_ids = tok.encode(request['target'], add_special_tokens=False)
        require(request['target'].startswith(' ') and len(target_ids) == 1, 'Frozen target assumptions')
        checks = []
        for i, template in enumerate(templates):
            text = template.format(request['prompt'])
            position = subject_last_reference_rule(tok, text, request['subject'])
            ids = tok.encode(text.format(request['subject']) + request['target'], add_special_tokens=False)[:-1]
            checks.append(position == m['positions'][i] and ids == m['rewrite_input_ids'][i] and target_ids == m['target_ids'][i])
        kl_position = subject_last_reference_rule(tok, '{} is a', request['subject'])
        require(all(checks) and kl_position == m['essence_position'], 'Prompt/position audit failed')
        require(tok.encode(request['subject'] + ' is a', add_special_tokens=False) == m['essence_input_ids'], 'KL row changed')
        keys = array(root / 'context_keys.npy'); mean = array(root / 'k_mean.npy')
        v = array(root / 'V0.npy').astype(np.float64)[:, 0]
        cv = c @ v; mean64 = mean.astype(np.float64)
        alpha = float(cv @ mean64 / (mean64 @ mean64))
        residual = float(np.linalg.norm(cv - alpha * mean64) / np.linalg.norm(cv))
        require(alpha > 0 and residual < 2e-5 and abs(np.linalg.norm(v) - 1) < 2e-5, 'R1 direction audit failed')
        base = array(root / 'w_k_star.npy'); values = array(root / 'context_native_values.npy')
        require(np.array_equal(keys[0], array(root / 'k_star.npy')), 'Constraint key is not clean')
        require(np.array_equal(base, values[0]), 'Base is not the first native value')
        require(np.array_equal(array(root / 'v_star.npy'), base + array(root / 'delta.npy')), 'Target is not base plus delta')
        history = m['history']; totals = trace_totals(history)
        require(len(history) == 20 and [h['step'] for h in history] == list(range(20)), 'R8 trace layout')
        require(m['gradient_steps'] == 20 and m['forwards'] == 462, 'R8 exact counts')
        base_norm = float(np.linalg.norm(base.astype(np.float64)))
        decay_errors = []
        for i, h in enumerate(history):
            before = 0. if i == 0 else history[i - 1]['delta_norm']
            predicted = .5 * before / base_norm**2
            decay_errors.append(abs(predicted - h['decay']))
            require(np.isclose(predicted, h['decay'], rtol=2e-5, atol=2e-5), 'R7 decay audit failed')
            require(h['delta_norm'] <= 4 * base_norm + 2e-5, 'Clamp bound exceeded')
        final = history[-1]
        result['facts'][fact] = dict(source_sha256=m['source_sha256'],
            positions_and_ids_match_reference_rule=True, rewrite_rows_checked=21, kl_rows_checked=1,
            V_norm=float(np.linalg.norm(v)), relative_CV_direction_residual=residual,
            clean_base_and_target_exact=True, maximum_decay_recalculation_error=max(decay_errors),
            loss_minimum=min(totals), loss_last=totals[-1], all_losses_above_stop_threshold=all(t >= .05 for t in totals),
            kl_first_already_weighted=history[0]['kl'],
            loss_last_if_KL_erroneously_weighted_again=final['nll'] + .0625 * final['kl'] + final['decay'],
            final_fields=final, counts=dict(forwards=m['forwards'], gradient_steps=m['gradient_steps']),
            affine_constraint=frozen['cases'][fact]['algebra_codex'])
    # This is a source review, rather than a claim that every equation was observed anew.
    rules = [
        ('R1', 'MATCH_MATH_NUMERICAL_CONVENTION', 'p1_gpt2_solver.py:93-98', 'Mean over captured keys; float64 Cholesky solve and normalization, cast V to float32. Not the reference float32 inverse/matmul path.'),
        ('R2', 'MATCH_FOR_FIVE_FROZEN_TARGETS', 'p1_gpt2_solver.py:26-47,78-79,103-106', '21 rewrite rows plus one KL row, right padding. All five targets are single tokens; 110 subject lookups agree with the reference token-length rule.'),
        ('R3', 'MATCH_MATH_NUMERICAL_CONVENTION', 'p1_gpt2_solver.py:57-65,100-106', 'Zero float32 delta added on every row, including KL. Codex casts delta to output dtype before adding; the reference uses mixed-dtype in-place addition.'),
        ('R4', 'MATCH_VALUE_CAPTURE_TIMING_DIFFERENCE', 'p1_gpt2_solver.py:81-100', 'The first clean native c_proj output is captured in the separate key extraction, then promoted to float32; not captured from the first 22-row optimizer batch.'),
        ('R5', 'MATCH_MATH_NUMERICAL_CONVENTION', 'p1_gpt2_solver.py:50-54,107-111', 'Final transformer output already includes ln_f, then tied head. Cross-entropy on float32 logits, target mean per row, then mean of 21 rows.'),
        ('R6', 'MATCH_KL_EQUATION_BASELINE_CAPTURE_DIFFERENCE', 'p1_gpt2_solver.py:91-92,112-113', 'KL(edited || native), coefficient .0625, subject_last on KL row. Native baseline is a separate singleton forward before optimization, not the first optimizer batch. Log-softmax is float32.'),
        ('R7', 'MATCH_MATH_NUMERICAL_CONVENTION', 'p1_gpt2_solver.py:100-114', 'Unsquared delta norm over clean-base norm squared, coefficient .5. Norms and penalty use float32. Stored decay is already weighted.'),
        ('R8', 'EXPLICIT_A3_EXCEPTION', 'p1_gpt2_solver.py:102-125', 'Adam lr .5, exactly 20 actual updates, no early stop, clamp after each update. Loss terms are before the update; delta_norm is after it.'),
        ('R9', 'MATCH_MATH_NUMERICAL_CONVENTION', 'p1_gpt2_solver.py:68-71,126-131', 'Clean affine constraint including bias. Dot product is float64; target/base and saved U are float32. Each saved constraint independently passes.'),
        ('R10', 'MATCH_DECLARED_ADAPTER_FORMAT', 'p1_gpt2_solver.py:145-155', 'U has shape (1600,1), V (6400,1), scale1, c_proj Conv1D delta_W=V@U.T; stored as the approved low_rank adapter.'),
        ('R11', 'MATCH_FOR_FIVE_FROZEN_TARGETS', 'run_p1_solve_codex.py:12-17,28-32', 'Leading spaces are present in the five fixed requests and checked against the qualification specs before execution.')]
    result['rule_review'] = {name: dict(status=status, source='tools/rome_solver/' + location, finding=finding) for name, status, location, finding in rules}
    result['cpu_rounding_example'] = rounding_example()
    result['final_loss_range'] = [min(f['loss_last'] for f in result['facts'].values()), max(f['loss_last'] for f in result['facts'].values())]
    result['A3_budget_arithmetic'] = dict(bench_caps=[832,784,784,832,832], bench_sum=4064, probe_total_cap=270,
        p1_5_cap=4334, spent_since_v2=6720, new_solves_expected=2310,
        since_v2_total=6720+2310+4334, including_reported_v1=6720+2310+4334+500,
        scope='A3 table arithmetic checked; plan.items not independently rerun in this audit')
    result['findings_before_approval'] = [
        'The mathematical rules align, with the explicit R8 exception, but R1/R3/R4/R5/R6/R7/R9 are not a statement of identical numerical execution. Declare these conventions before the new comparison; keep the frozen Codex code unchanged.',
        'Correct the trace-loss range to the sum of already weighted terms. The 0.115-0.187 range is reproduced by weighting KL a second time. This numerical reporting error does not change the no-early-stop observation on the archived Codex trajectories.',
        'Native KL capture is separate in Codex. A3 may keep a reference-style first-iteration baseline in Claude, but must disclose the difference and retain the existing comparison thresholds. No new real pass was made to measure its impact.'
    ]
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(rules=len(rules), subject_rows_checked=110, final_loss_range=result['final_loss_range'], budget=result['A3_budget_arithmetic'], scalar_rounding_example=result['cpu_rounding_example'])))


if __name__ == '__main__':
    main()
