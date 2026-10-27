"""Model-free tests of the D1 diagnostic (docs/phase-b-d1-diagnostic-protocol.md)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import phase_b_d1_diagnostic as d1  # noqa: E402


def test_first_value_is_earliest_and_word_bounded():
    assert d1.first_value('Hyundai is a **South Korean** company.', ['South Korea', 'Korean', 'Korea']) == (1, 'Korean', 21)
    assert d1.first_value('Toyota comes from **Japan**.', ['Japan', 'Japanese']) == (0, 'Japan', 20)
    assert d1.first_value('Germanic tribes', ['German']) is None


def test_pair_new_aligns_equal_lists_and_matches_case():
    jp, de = ['Japan', 'Japanese'], ['Germany', 'German']
    assert d1.pair_new(1, 'Japanese', jp, de) == 'German'
    assert d1.pair_new(0, 'Japan', jp, de) == 'Germany'
    sprint, swim = ['sprint', 'sprinting', 'Sprinting', 'athletics', 'Athletics', 'track', 'running'], ['swimming', 'Swimming', 'swimmer']
    assert d1.pair_new(5, 'track', sprint, swim) == 'swimming'
    assert d1.pair_new(2, 'Sprinting', sprint, swim) == 'Swimming'


def test_probe_item_prefix_and_exclusions():
    spec = dict(old=dict(accept=['Japan', 'Japanese']), new=dict(accept=['Germany', 'German']))
    e = dict(id='E1', control='efficacy')
    assert d1.probe_item(e, spec, 'Toyota comes from **Japan**.', ['Toyota']) == dict(
        qid='E1', prefix='Toyota comes from **', true='Japan', new='Germany')
    vw = dict(id='O2', control='off_target', expect=dict(accept=['Germany', 'German']))
    assert 'excluded' in d1.probe_item(vw, spec, 'Volkswagen is a German company.', ['Volkswagen'])
    assert 'excluded' in d1.probe_item(e, spec, 'It is a car company.', ['Toyota'])
    kenya = dict(id='O1', control='off_target', expect=dict(accept=['Kenya']))
    chile = dict(old=dict(accept=['Tanzania']), new=dict(accept=['Chile']))
    assert d1.probe_item(kenya, chile, 'Mount Kenya is located in **Kenya**.', ['Mount Kenya'])['prefix'] == \
        'Mount Kenya is located in **'


def test_auc_and_spearman():
    assert d1.auc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]) == 1.0
    assert d1.auc([0.1, 0.2, 0.3], [0.9, 0.8, 0.7]) == 0.0
    assert d1.auc([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.5
    assert d1.auc([1, 2], [0, 0, 0]) is None
    assert d1.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert d1.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_efficacy_classes():
    both = {d1.AFTER: 1, d1.BEFORE: 2}
    assert d1.efficacy_class(1.0, 5, both, c1_pass=True) == 'P'
    assert d1.efficacy_class(0.3, 5, {}, c1_pass=False) == 'K'
    assert d1.efficacy_class(0.9, 1.0, {}, c1_pass=False) == 'V'
    assert d1.efficacy_class(0.9, 3.0, both, c1_pass=False) == 'S'
    assert d1.efficacy_class(0.9, 3.0, {d1.AFTER: 2}, c1_pass=False) == 'U'


def test_arm_effect_rules():
    c = [0.6, 0.6, 0.6, 0.6]
    assert d1.arm_effect([0.3, 0.3, 0.3, 0.6], c, [True] * 4) == 'improves'
    assert d1.arm_effect([0.3, 0.3, 0.3, 0.6], c, [True, False, False, True]) == 'neutral'   # efficacy lost
    assert d1.arm_effect([0.9, 0.9, 0.95, 0.1], c, [True] * 4) == 'worse'
    assert d1.arm_effect([0.5, 0.55, 0.6, 0.7], c, [True] * 4) == 'neutral'


def test_wiki_records_split_at_top_level_headings():
    raw = ' = Alpha = \n text a \n = = Sub = = \n more a \n = Beta = \n text b \n'
    recs = d1.wiki_records(raw)
    assert [r[0] for r in recs] == ['00000', '00001']
    assert 'Sub' in recs[0][1] and recs[1][1].startswith(' = Beta = ')


def test_subject_last_token_uses_longest_surface():
    text = '<user>The Eiffel Tower is located in which city?'
    offsets = [(0, 6), (6, 9), (9, 16), (16, 22), (22, 25), (25, 33)]
    assert d1.subject_last_token(offsets, text, ['Eiffel Tower', 'The Eiffel Tower']) == 3
    with pytest.raises(ValueError):
        d1.subject_last_token(offsets, text, ['Louvre'])


def _edit(tmp, layer, rng, dim=8):
    e = tmp / f'L{layer}'
    e.mkdir()
    k = rng.standard_normal(dim)
    M = np.eye(dim) + 0.1 * np.ones((dim, dim))
    q = np.linalg.solve(M, k)
    np.save(e / 'k_star.npy', k.astype(np.float32))
    np.save(e / 'V0.npy', (q / (k @ q)).astype(np.float32)[:, None])
    return e


def test_arm_v_unit_dose_at_k_star(tmp_path):
    rng = np.random.default_rng(0)
    e = _edit(tmp_path, 8, rng)
    k = np.load(e / 'k_star.npy').astype(np.float64)
    (tmp_path / 'covariance').mkdir()
    np.save(tmp_path / 'covariance' / 'W8.npy', (np.eye(8) * 2.0).astype(np.float32))
    for arm in ('C', 'I', 'W'):
        assert d1.arm_v(arm, e, 8, tmp_path) @ k == pytest.approx(1.0, rel=1e-5)


def test_analyze_end_to_end_on_synthetic_records(tmp_path):
    """A plan with one fact and one edit whose neighbour doses line up with the bleed labels."""
    rng = np.random.default_rng(1)
    e = _edit(tmp_path, 8, rng)
    k = np.load(e / 'k_star.npy').astype(np.float64)
    V = np.load(e / 'V0.npy').astype(np.float64)[:, 0]
    keys, pos = {}, {}
    for q in d1.M1_PROMPTS:
        x = rng.standard_normal((5, 8))
        if q in d1.UNSEEN or q == 'S1':
            x[2] = k                                  # full dose at the subject
        elif q in ('O1', 'O2'):
            x[2] = 0.7 * k + 0.01 * rng.standard_normal(8)
        else:
            x[2] = 0.0
        keys[f'{q}:L8'] = x
        pos[q] = None if q == 'W1' else 2
    def res(gain):
        return {q: dict(true=[-1.0], new=[-1.0 + gain.get(q, 0.0)]) for q in d1.M2_PROMPTS}
    margins = {'native': res({q: -3.0 for q in d1.M2_PROMPTS}),
               'C:L8': res({**{q: 4.0 for q in d1.UNSEEN}, 'S1': 4.0, 'O1': 2.0, 'O2': 2.0, 'O3': -3.0, 'O4': -3.0}),
               'I:L8': res({**{q: 4.0 for q in d1.UNSEEN}, 'S1': 4.0, 'O1': 1.0, 'O2': 1.0, 'O3': -3.0, 'O4': -3.0}),
               'random:L8': res({q: -3.0 for q in d1.M2_PROMPTS})}
    b2 = dict(statuses=dict(C1='PASS', C6='FAIL'), bleed=dict(O1=True, O2=True, O3=False, O4=False))
    svc = dict(kind='fact', fact_id='f', forwards=0, edits=[dict(layer=8, edit=str(e), b2=b2)])
    plan = dict(protocol='p', arms=['C', 'I'], services=[svc])
    plan['plan_sha256'] = d1.sha_json(plan)
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    s = tmp_path / 'services' / '0'
    s.mkdir(parents=True)
    np.savez(s / 'keys.npz', **keys)
    m3 = {'L8': dict(native_rel=0.001, C=dict(rel=0.01), I=dict(rel=0.01))}
    (s / 'records.json').write_text(json.dumps(dict(positions=pos, margins=margins, m3=m3)))
    out = d1.analyze(tmp_path)
    row = out['rows'][0]
    assert row['arms']['C']['dose_E'] == pytest.approx(1.0, rel=1e-5)
    assert row['arms']['C']['dose']['O1'] == pytest.approx(0.7, abs=0.05)
    assert row['arms']['C']['class'] == 'P' and out['probe_valid'] and out['m3_ok']
    assert out['bleed_auc_C'] is None                 # too few pairs of each label for an AUC
    assert row['bleed_b2'] == dict(O1=True, O2=True, O3=False)
    assert row['arms']['I']['dose']['S1'] == pytest.approx(1.0, rel=1e-5)
    assert out['random_noise_ok']


def test_plan_on_the_real_inputs_respects_the_service_cap():
    if not d1.B2V2.exists() or not d1.SOLVES.exists():
        pytest.skip('B2 v2 records or solves not on this machine')
    for wiki in (False, True):
        plan = d1.build_plan(wiki)
        facts = [s for s in plan['services'] if s['kind'] == 'fact']
        assert len(facts) == 5 and all(s['forwards'] <= d1.FORWARD_CAP for s in plan['services'])
        assert plan['forwards_total'] <= (1420 if wiki else 950)
        assert all(len(s['conditions']) == (13 if wiki else 10) for s in facts)
