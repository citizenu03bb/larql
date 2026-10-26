"""Replay adapters on synthetic archives, and (when present) the exit criterion on the real archives."""
import json

import numpy as np
import pytest

from edit_cert.replay import replay, replay_all
from edit_cert.replay.archives import CASES, DEFAULT_ROOT, EXPECTED
from replay_fixtures import build


@pytest.fixture
def archives(tmp_path):
    return build(tmp_path / 'archives')


@pytest.mark.parametrize('case', list(CASES))
def test_fixture_replay_matches_expected(archives, case):
    v, rep = replay(case, archives)
    assert rep['ok'], rep
    assert v['verdict'] == 'REFUSED' and not v['synthetic']


def test_replay_writes_reports(archives, tmp_path):
    s = replay_all(archives, tmp_path / 'out')
    assert s['exit_criterion_met']
    assert (tmp_path / 'out' / 'replay-report.md').exists()
    assert json.loads((tmp_path / 'out' / 'R2-verdict.json').read_text())['verdict'] == 'REFUSED'


def test_a_changed_archived_number_is_caught(archives):
    p = archives / 'nf4-phase3b-rows-20261026' / 'verification.json'
    raw = json.loads(p.read_text())
    raw['results']['number']['row30']['increment'] += 1e-6
    p.write_text(json.dumps(raw))
    _, rep = replay('R2', archives)
    assert not rep['ok'] and rep['unreproduced']


def test_a_changed_artifact_is_refused(archives):
    d = archives / 'nf4-asymmetry-20261026' / 'interventions'
    f = next(d.glob('*-symbol_residual.npz'))
    np.savez_compressed(f, logits=np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError):
        replay('R1', archives)


def test_expected_table_covers_every_mandatory_control():
    from edit_cert.controls import MANDATORY
    for exp in EXPECTED.values():
        assert {'verdict', *MANDATORY} <= set(exp)


@pytest.mark.skipif(not (DEFAULT_ROOT / 'nf4-asymmetry-20261026').exists(), reason='private archives not present')
def test_exit_criterion_on_real_archives():
    s = replay_all(DEFAULT_ROOT)
    assert s['exit_criterion_met'], {k: (r['mismatch'], r['unreproduced']) for k, r in s['cases'].items()}


def test_amended_coverage(archives):
    v1, _ = replay('R1', archives)
    c5 = next(c for c in v1['controls'] if c['id'] == 'C5')
    assert c5['status'] == 'NOT_EVALUATED' and c5['data']['observed_on_seen']['SYM']
    v3, _ = replay('R3', archives)
    assert v3['qualification']['LI']['level'] == 'boundary' and v3['qualification']['C']['level'] == 'boundary'
