"""Full pipeline (prepare -> services -> verdict) on the fake instrument with planted edits."""
import json
from pathlib import Path

import pytest

from edit_cert import plan as planner, runner
from edit_cert.backends.fake import FakeEdit
from edit_cert.instruments.fake import FakeInstrument, Override
from edit_cert.spec import SpecError, load

SPEC = Path(__file__).with_name('fake-hydrogen-symbol.json')
SEEN = ["Give hydrogen's chemical symbol."]


def certify(tmp_path, edit, name='run'):
    edit.save(tmp_path / f'{name}-edit')
    plan = runner.prepare(tmp_path / name, SPEC, tmp_path / f'{name}-edit', 'fake')
    inst = FakeInstrument()
    for k in range(len(plan['services'])):
        runner.run_service(tmp_path / name, k, inst)
    v = runner.certify(tmp_path / name)
    return v, {c['id']: c['status'] for c in v['controls']}


def refused_by(v):
    return {r['control'] for r in v['reasons']}


def test_selective_edit_is_certified(tmp_path):
    v, st = certify(tmp_path, FakeEdit([Override('hydrogen', 'symbol', 'He')], seen_phrasings=SEEN))
    assert v['verdict'] == 'CERTIFIED' and v['synthetic'] is True
    assert st['C5'] == 'PASS'
    assert (tmp_path / 'run' / 'verdict.md').read_text().startswith('# Edit certification: CERTIFIED')


@pytest.mark.parametrize('edit,expected', [
    (FakeEdit([Override('hydrogen', 'symbol', 'He'), Override('hydrogen', 'number', '3')]), {'C2', 'C3'}),
    (FakeEdit([Override('hydrogen', 'symbol', 'He', when='subject_before_relation')]), {'C1', 'C3', 'C4'}),
    (FakeEdit([]), {'C1', 'C3', 'C4'}),
    (FakeEdit([Override('hydrogen', 'symbol', 'He')], persists=False), {'C7'}),
    (FakeEdit([Override('hydrogen', 'symbol', 'He')], one_shot=True), {'C0', 'C1', 'C3'}),
    (FakeEdit([Override('hydrogen', 'symbol', 'He', rename='helium')]), {'C1', 'C2', 'C3'}),
    (FakeEdit([Override('hydrogen', 'symbol', 'He')], form=dict(max_write_layer=30)), {'C0'}),
], ids=['leaks-attribute', 'subject-before-only', 'inert', 'lost-on-reload', 'one-shot', 'entity-swap',
        'after-kv-deadline'])
def test_planted_failures_are_refused_for_the_right_reasons(tmp_path, edit, expected):
    v, _ = certify(tmp_path, edit)
    assert v['verdict'] == 'REFUSED'
    assert refused_by(v) == expected


def test_off_target_leak_is_refused(tmp_path):
    v, st = certify(tmp_path, FakeEdit([Override('hydrogen', 'symbol', 'He'), Override('lithium', 'symbol', 'Xx')]))
    assert st['C6'] == 'FAIL' and refused_by(v) == {'C6'}


def test_provenance_is_checked_at_prepare(tmp_path):
    edit = FakeEdit([Override('hydrogen', 'symbol', 'He')], seen_phrasings=['What is the chemical symbol for hydrogen?'])
    edit.save(tmp_path / 'e')
    with pytest.raises(SpecError):
        runner.prepare(tmp_path / 'run', SPEC, tmp_path / 'e', 'fake')


def test_services_respect_the_cap_and_phases():
    spec = load(SPEC)
    services = planner.pack(planner.items(spec))
    assert all(sum(i.cost for i in s) <= planner.SERVICE_CAP for s in services)
    phases = [s[0].phase for s in services]
    assert phases == sorted(phases, key=['native', 'edit', 'reload'].index)
    assert all(len({i.phase for i in s}) == 1 for s in services)
    with pytest.raises(ValueError):
        planner.pack(planner.items(spec), cap=5)


def test_budget_refuses_extra_forwards():
    inst = FakeInstrument()
    b = runner.Budget(inst, cap=1)
    b.forward(inst.chat_prompt('What is the capital of France?'))
    with pytest.raises(RuntimeError):
        b.forward([1, 2])


def test_services_run_in_order_and_once(tmp_path):
    FakeEdit([Override('hydrogen', 'symbol', 'He')]).save(tmp_path / 'e')
    runner.prepare(tmp_path / 'run', SPEC, tmp_path / 'e', 'fake')
    inst = FakeInstrument()
    with pytest.raises(ValueError):
        runner.run_service(tmp_path / 'run', 1, inst)
    runner.run_service(tmp_path / 'run', 0, inst)
    with pytest.raises(FileExistsError):
        runner.run_service(tmp_path / 'run', 0, inst)


def test_tampering_is_detected(tmp_path):
    v, _ = certify(tmp_path, FakeEdit([Override('hydrogen', 'symbol', 'He')]))
    rec = tmp_path / 'run' / 'services' / '0' / 'records.json'
    rec.write_text(rec.read_text().replace('"H"', '"X"', 1) + ' ')
    with pytest.raises(ValueError):
        runner.certify(tmp_path / 'run')
    plan_path = tmp_path / 'run' / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['cap'] = 99
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        planner.load(tmp_path / 'run')


def test_selftest_command(tmp_path):
    from edit_cert.__main__ import main
    assert main(['selftest', '--out', str(tmp_path / 'st')]) == 0
