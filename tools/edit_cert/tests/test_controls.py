"""Control and verdict logic on hand-built evidence (no instrument at all)."""
import pytest

from edit_cert import controls as C
from edit_cert.evidence import Boundary, Evidence, Generation
from edit_cert.spec import SpecError, parse
from edit_cert.verdict import decide, render_md

FORM = dict(answer_time_trigger=False, donor_pass=False, uses_native_proposal=False,
            substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=True, max_write_layer=8)

RAW = {
    'fact_id': 't', 'subject': ['hydrogen'], 'relation': 'symbol', 'old': {'accept': ['H']}, 'new': {'accept': ['He']},
    'swap_markers': ['helium'], 'min_efficacy': 2,
    'questions': [
        {'id': 'S', 'control': 'efficacy', 'order': 'subject_before_relation', 'text': 'seen', 'seen': True},
        {'id': 'E1', 'control': 'efficacy', 'order': 'subject_after_relation', 'text': 'e1'},
        {'id': 'E2', 'control': 'efficacy', 'order': 'subject_before_relation', 'text': 'e2'},
        {'id': 'A1', 'control': 'attribute', 'order': 'subject_after_relation', 'text': 'a1',
         'expect': {'accept': ['1'], 'reject': ['2']}},
        {'id': 'A2', 'control': 'attribute', 'order': 'subject_before_relation', 'text': 'a2',
         'expect': {'accept': ['1'], 'reject': ['2']}},
        {'id': 'W', 'control': 'answer_only', 'order': 'subject_in_answer_only', 'text': 'w'},
        {'id': 'O', 'control': 'off_target', 'order': 'none', 'text': 'o'},
    ]}
SPEC = parse(RAW)

NATIVE = {'S': "Hydrogen's symbol is H.", 'E1': 'It is H.', 'E2': "Hydrogen's symbol is H.", 'A1': 'It is 1.',
          'A2': 'Number 1.', 'W': 'Hydrogen and oxygen.', 'O': 'Paris.'}
GOOD = {'S': 'He.', 'E1': 'It is He.', 'E2': 'Symbol He.', 'A1': 'It is 1.', 'A2': 'Number 1.',
        'W': 'Hydrogen and oxygen.', 'O': 'Paris.'}


def gen(qid, cond, text, eos=True):
    return Generation(qid, cond, tuple(ord(c) for c in text), text, eos, not eos)


def evidence(edited=None, native=None, reload=True, random=True, ratio=(20.0, 0.0), drop=()):
    ev = Evidence('test', True, 22)
    for qid, text in {**NATIVE, **(native or {})}.items():
        if qid not in drop:
            ev.generations.append(gen(qid, 'native', text))
    for qid, text in {**GOOD, **(edited or {})}.items():
        if qid not in drop:
            ev.generations.append(gen(qid, 'edit', text if text is not None else 'cut', eos=text is not None))
    for qid in ('E1', 'E2'):
        if random:
            ev.generations.append(gen(qid, 'random', NATIVE[qid]))
        if reload:
            ev.generations.append(gen(qid, 'reloaded', {**GOOD, **(edited or {})}[qid] or 'cut'))
        if ratio:
            ev.boundaries += [Boundary(qid, 'native', 1, 2, 1, -10.0, 'all'),
                              Boundary(qid, 'edit', 1, 2, 2, -10.0 + ratio[0], 'all'),
                              Boundary(qid, 'random', 1, 2, 1, -10.0 + ratio[1], 'all')]
    ev.artifact_sha256 = {'edit': 'x', 'reloaded': 'x'} if reload else {'edit': 'x'}
    return ev


def status(ev, form=FORM, spec=SPEC):
    v = decide(spec, form, ev)
    return v['verdict'], {c['id']: c['status'] for c in v['controls']}


def test_clean_evidence_is_certified():
    verdict, st = status(evidence())
    assert verdict == 'CERTIFIED'
    assert all(st[c] == 'PASS' for c in C.MANDATORY)


def test_attribute_moved_fails_c2_and_c3():
    verdict, st = status(evidence(edited={'A1': 'It is 2.'}))
    assert verdict == 'REFUSED' and st['C2'] == 'FAIL' and st['C3'] == 'FAIL'


def test_order_dependent_edit_fails_c1_and_c3():
    _, st = status(evidence(edited={'E1': 'It is H.'}))
    assert st['C1'] == 'FAIL' and st['C3'] == 'FAIL'


def test_swap_marker_fails_efficacy():
    _, st = status(evidence(edited={'E1': 'helium is He.'}))
    assert st['C1'] == 'FAIL'


def test_cap_hit_is_not_a_pass():
    _, st = status(evidence(edited={'E1': None}))
    assert st['C1'] == 'NOT_EVALUATED'


def test_wrong_native_answer_excludes_the_question():
    v = decide(SPEC, FORM, evidence(native={'E1': 'It is X.'}))
    assert v['qualification']['E1']['level'] == 'excluded'
    st = {c['id']: c['status'] for c in v['controls']}
    assert st['C1'] == 'NOT_EVALUATED'   # one unseen phrasing left, below min_efficacy=2


def test_empty_order_group_is_not_evaluated():
    _, st = status(evidence(drop=('A2',)))
    assert st['C3'] == 'NOT_EVALUATED' and st['C2'] == 'PASS'


def test_answer_only_swap_fails_c3():
    _, st = status(evidence(edited={'W': 'Helium and oxygen.'}))
    assert st['C3'] == 'FAIL'


def test_random_that_flips_fails_c4():
    ev = evidence()
    ev.generations = [g for g in ev.generations if g.condition != 'random'] + [gen('E1', 'random', 'He.'),
                                                                               gen('E2', 'random', 'H.')]
    _, st = status(ev)
    assert st['C4'] == 'FAIL'


def test_weak_gain_ratio_fails_c4_and_missing_ratio_is_not_evaluated():
    assert status(evidence(ratio=(4.0, 3.0)))[1]['C4'] == 'FAIL'
    assert status(evidence(ratio=None))[1]['C4'] == 'NOT_EVALUATED'


def test_off_target_change_fails_c6():
    assert status(evidence(edited={'O': 'Lyon.'}))[1]['C6'] == 'FAIL'


def test_reload_mismatch_and_missing_reload():
    ev = evidence()
    ev.generations = [g for g in ev.generations if not (g.condition == 'reloaded' and g.question_id == 'E1')]
    ev.generations.append(gen('E1', 'reloaded', 'It is H.'))
    assert status(ev)[1]['C7'] == 'FAIL'
    assert status(evidence(reload=False))[1]['C7'] == 'NOT_EVALUATED'
    ev = evidence()
    ev.artifact_sha256['reloaded'] = 'y'
    assert status(ev)[1]['C7'] == 'FAIL'


@pytest.mark.parametrize('key,value', [('answer_time_trigger', True), ('donor_pass', True),
                                       ('uses_native_proposal', True), ('applied_every_forward', False),
                                       ('max_write_layer', 23)])
def test_form_failures(key, value):
    assert status(evidence(), form=dict(FORM, **{key: value}))[1]['C0'] == 'FAIL'


def test_form_missing_or_deadline_unknown():
    form = {k: v for k, v in FORM.items() if k != 'donor_pass'}
    assert status(evidence(), form=form)[1]['C0'] == 'NOT_EVALUATED'
    ev = evidence()
    ev.kv_write_deadline = None
    assert status(ev)[1]['C0'] == 'NOT_EVALUATED'


def test_boundary_can_refuse_but_not_certify():
    ev = Evidence('test', False, 22)
    for qid in ('E1', 'E2', 'A1', 'A2'):
        ev.boundaries.append(Boundary(qid, 'native', 1, 2, 1, -10.0, 'all'))
    ev.boundaries += [Boundary('E1', 'edit', 1, 2, 2, 5.0, 'all'), Boundary('E2', 'edit', 1, 2, 2, 5.0, 'all'),
                      Boundary('A1', 'edit', 1, 2, 1, -9.0, 'all'), Boundary('A2', 'edit', 1, 2, 1, -9.0, 'all')]
    st = status(ev)[1]
    assert st['C1'] == 'NOT_EVALUATED' and st['C2'] == 'NOT_EVALUATED'
    ev.boundaries.append(Boundary('A1', 'edit', 1, 2, 2, 3.0, 'all', variant='second'))
    assert status(ev)[1]['C2'] == 'FAIL'


def test_generation_decides_over_its_own_boundary():
    ev = evidence()
    ev.boundaries.append(Boundary('E1', 'edit', 1, 2, 1, -5.0, 'all', variant='no flip on the native prefix'))
    assert status(ev)[1]['C1'] == 'PASS'


def test_seen_phrasing_can_refuse_but_not_pass():
    assert status(evidence(edited={'S': 'H.'}))[1]['C1'] == 'FAIL'


def test_row_split_attributes_feed_c3_by_mention():
    ev = evidence()
    ev.boundaries += [Boundary('A2', 'native', 1, 2, 1, -10.0, 'all'),
                      Boundary('A2', 'edit', 1, 2, 2, 1.0, 'subject_after_relation', partial=True)]
    res = {c['id']: c for c in decide(SPEC, FORM, ev)['controls']}
    assert res['C3']['status'] == 'FAIL' and 'subject_after_relation' in res['C3']['detail']


def test_donor_that_cannot_flip_removes_the_phrasing():
    ev = evidence()
    ev.boundaries.append(Boundary('E1', 'donor', 1, 2, 1, -3.0, 'all'))
    res = {c['id']: c for c in decide(SPEC, FORM, ev)['controls']}
    assert res['C5']['status'] == 'FAIL' and res['C1']['status'] == 'NOT_EVALUATED'


def test_markdown_flags_synthetic_and_lists_reasons():
    v = decide(SPEC, dict(FORM, donor_pass=True), evidence())
    md = render_md(v)
    assert 'SYNTHETIC' in md and '**C0**' in md and v['verdict'] == 'REFUSED'


@pytest.mark.parametrize('bad', [
    {'questions': [{'id': 'x', 'control': 'nope', 'order': 'none', 'text': 't'}]},
    {'questions': [{'id': 'x', 'control': 'attribute', 'order': 'none', 'text': 't'}]},
    {'questions': [{'id': 'x', 'control': 'off_target', 'order': 'sideways', 'text': 't'}]},
    {'questions': [{'id': 'x', 'control': 'off_target', 'order': 'none', 'text': 't'},
                   {'id': 'x', 'control': 'off_target', 'order': 'none', 'text': 'u'}]},
])
def test_spec_validation(bad):
    raw = {'fact_id': 't', 'subject': ['h'], 'relation': 'r', 'old': {'accept': ['a']}, 'new': {'accept': ['b']}, **bad}
    with pytest.raises(SpecError):
        parse(raw)


def test_provenance_rejects_a_seen_text_marked_unseen():
    with pytest.raises(SpecError):
        SPEC.check_provenance(['  E1 '])
    SPEC.check_provenance(['seen'])


def test_donor_on_a_seen_phrasing_is_reported_not_qualifying():
    ev = evidence()
    ev.boundaries.append(Boundary('S', 'donor', 1, 2, 2, 3.0, 'all'))
    res = {c['id']: c for c in decide(SPEC, FORM, ev)['controls']}
    assert res['C5']['status'] == 'NOT_EVALUATED' and 'seen' in res['C5']['detail']
    assert res['C1']['status'] == 'PASS'


def test_answer_only_rejected_value():
    spec = parse({**{k: v for k, v in RAW.items() if k != 'questions'},
                  'questions': RAW['questions'][:-2] + [
                      {'id': 'W', 'control': 'answer_only', 'order': 'subject_in_answer_only', 'text': 'w',
                       'expect': {'reject': ['Rome']}},
                      RAW['questions'][-1]]})
    assert status(evidence(edited={'W': 'Hydrogen in Rome.'}), spec=spec)[1]['C3'] == 'FAIL'
    assert status(evidence(), spec=spec)[1]['C3'] == 'PASS'
