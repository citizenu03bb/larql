"""Phase B2 certification driver (Claude Code), under docs/phase-b2-certification-protocol.md.

  python3 scripts/phase_b2_certify.py [--v2] --rehearse           # model-free rehearsal, no service started
  python3 scripts/phase_b2_certify.py [--v2] --run --i-have-approval

--v2 (docs/phase-b2-v2-recertification-protocol.md): v2 specs (64 tokens, value-based off-target, subject-naming
answer-only), a new certification root, and a native feasibility gate run before any edited service.

For each solve listed `complete` in Codex's handoff inventory, in inventory order:
  1. the independent algebra check (xcheck_rome_algebra.py) must pass; a failure is recorded, never repaired;
  2. a minimal copy of the edit (backend.json, U0, V0) is planned at --cap 400 on nf4-e2b with the public spec;
  3. its services run one at a time through run_service.sh (shared flock, systemd-run limits);
  4. the model-free verdict is written, and the case enters the index.
Any failed service stops the whole series (no retry). Cases still pending are awaited until all are complete or
failed, or the wall-clock limit is reached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # tools/edit_cert
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'scripts'))
from edit_cert import plan as planner, runner  # noqa: E402
import xcheck_rome_algebra  # noqa: E402

SOLVE_ROOT = Path.home() / 'Projects/larql-research/phase-b-rome-20261026'
CERT_ROOT = Path.home() / 'Projects/larql-research/phase-b-certification-20261026'
INVENTORY = SOLVE_ROOT / 'certification-handoff-latest.json'
PROTOCOL = 'docs/phase-b2-certification-protocol.md'
SPECS = HERE / 'facts' / 'phase-b-dev'
V2 = False


def use_v2():
    global CERT_ROOT, PROTOCOL, SPECS, V2
    CERT_ROOT = Path.home() / 'Projects/larql-research/phase-b-certification-v2-20261026'
    PROTOCOL = 'docs/phase-b2-v2-recertification-protocol.md'
    SPECS = HERE / 'facts' / 'phase-b-dev-v2'
    V2 = True
MINIMAL = ('backend.json', 'U0.npy', 'V0.npy')
WALL_LIMIT = 8 * 3600


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p: Path, v) -> None:
    p.write_text(json.dumps(v, indent=1, sort_keys=True) + '\n')


def case_name(c: dict) -> str:
    return f"{c['fact_id']}-L{c['layer']}"


def prepare_case(c: dict, root: Path) -> dict:
    name = case_name(c)
    d = root / name
    d.mkdir(parents=True, exist_ok=False)
    alg = xcheck_rome_algebra.check(Path(c['folder']), SOLVE_ROOT / 'b1')
    write(d / 'algebra-xcheck-claude.json', alg)
    if not alg['ok']:
        return dict(case=name, state='algebra_failed', failed=[k for k, v in alg['checks'].items() if not v.get('ok', True)])
    spec = SPECS / f"{c['fact_id']}.json"
    if not V2:   # v2 uses the amended specs; the solves did not depend on them (only S1, contexts, essence prompt)
        assert sha(spec) == c['spec_sha256'], f'{spec}: hash differs from the handoff'
    edit = d / 'edit-min'
    edit.mkdir()
    for f in MINIMAL:
        shutil.copyfile(Path(c['edit']) / f, edit / f)
    plan = runner.prepare(d / 'run', spec, edit, 'nf4-e2b', cap=400)
    assert plan['backend']['content_sha256'] == c['edit_content_sha256']
    return dict(case=name, state='planned', services=[s['forwards'] for s in plan['services']],
                plan_sha256=plan['plan_sha256'], algebra_ok=True)


def run_case(entry: dict, root: Path, log, only_phase: str | None = None) -> dict:
    run = root / entry['case'] / 'run'
    plan = planner.load(run)
    n = len(entry['services'])
    for k in range(n):
        if only_phase and plan['services'][k]['phase'] != only_phase:
            continue
        if (run / 'services' / str(k) / 'status.json').exists():
            continue
        cmd = ['bash', str(HERE / 'run_service.sh'), '--protocol', PROTOCOL, '--run', str(run),
               '--service', str(k), '--i-have-approval']
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        log(dict(case=entry['case'], service=k, returncode=proc.returncode, seconds=round(time.time() - t0, 1),
                 stdout_tail=proc.stdout[-400:], stderr_tail=proc.stderr[-800:]))
        if proc.returncode != 0 or not (run / 'services' / str(k) / 'status.json').exists():
            return dict(entry, state='service_failed', failed_service=k, returncode=proc.returncode)
    if only_phase:
        return dict(entry, state=f'{only_phase}_done')
    v = runner.certify(run)
    counts = [json.loads((run / 'services' / str(k) / 'status.json').read_text())['forwards'] for k in range(n)]
    st = {c['id']: c['status'] for c in v['controls']}
    return dict(entry, state='certified', verdict=v['verdict'], C1=st['C1'], C2=st['C2'], statuses=st,
                reasons=[r['control'] for r in v['reasons']], forwards=counts, forwards_total=sum(counts),
                verdict_sha256=sha(run / 'verdict.json'), evidence_sha256=sha(run / 'evidence.json'),
                records_sha256=[json.loads((run / 'services' / str(k) / 'status.json').read_text())['records_sha256']
                                for k in range(n)])


BEFORE, AFTER = 'subject_before_relation', 'subject_after_relation'


def native_coverage(run: Path) -> dict:
    """From native generations only: which unseen efficacy and attribute orders are generation-qualified."""
    from edit_cert import controls, spec as specs
    from edit_cert.evidence import Evidence, Generation
    sp = specs.load(run / 'spec.json')
    ev = Evidence('nf4-e2b', False)
    for st in sorted((run / 'services').glob('*/records.json')):
        r = json.loads(st.read_text())
        if r['phase'] == 'native':
            ev.generations += [Generation(**{**g, 'ids': tuple(g['ids'])}) for g in r['generations']]
    q = controls.qualify(sp, ev)
    def orders(ctl):
        return sorted({x.order for x in sp.questions if x.control == ctl and not x.seen and q[x.id][0] == 'generation'})
    eff = [x.id for x in sp.questions if x.control == 'efficacy' and not x.seen and q[x.id][0] == 'generation']
    cov = dict(efficacy_orders=orders('efficacy'), attribute_orders=orders('attribute'), efficacy_count=len(eff),
               excluded={k: v[1] for k, v in q.items() if v[0] == 'excluded'})
    cov['feasible'] = ({BEFORE, AFTER} <= set(cov['efficacy_orders']) and len(eff) >= sp.min_efficacy
                       and {BEFORE, AFTER} <= set(cov['attribute_orders']))
    return cov


def rehearse() -> int:
    """Model-free: algebra, minimal copy, real plans (in a scratch root), service wrapper refusals, fake run at 400."""
    import tempfile
    inv = json.loads(INVENTORY.read_text())
    tmp = Path(tempfile.mkdtemp(prefix='phase-b2-rehearsal-'))
    out = []
    for c in inv['cases']:
        if c['status'] == 'complete':
            e = prepare_case(c, tmp)
            assert e['state'] == 'planned' and all(f <= 400 for f in e['services']), e
            assert V2 or e['services'] == [312, 400, 24, 96], e
            out.append(e)
    r = subprocess.run(['bash', str(HERE / 'run_service.sh'), '--protocol', PROTOCOL, '--run', str(tmp), '--service', '0'],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 3 and 'no --i-have-approval' in r.stderr, r
    from edit_cert.__main__ import main
    assert main(['selftest', '--out', str(tmp / 'selftest')]) == 0
    print(json.dumps(dict(rehearsal='ok', planned=[e['case'] for e in out], scratch=str(tmp))))
    return 0


def run_all() -> int:
    CERT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = CERT_ROOT / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else dict(
        author='Claude Code', protocol=PROTOCOL, cases={})
    logf = open(CERT_ROOT / 'driver-log.jsonl', 'a')

    def log(rec):
        rec['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        logf.write(json.dumps(rec) + '\n')
        logf.flush()

    start = time.time()
    if V2 and 'feasibility' not in index:
        # native-only gate, one case per fact (the fact's first listed layer); its native services are reused there
        inv = json.loads(INVENTORY.read_text())
        firsts = {}
        for c in inv['cases']:
            firsts.setdefault(c['fact_id'], c)
        feas = {}
        for fact, c in firsts.items():
            entry = prepare_case(c, CERT_ROOT)
            log(dict(event='prepared', **entry))
            entry = run_case(entry, CERT_ROOT, log, only_phase='native')
            if entry['state'] == 'service_failed':
                index['stopped'] = f"native service failure in {entry['case']}"
                write(index_path, index)
                return 1
            feas[fact] = dict(case=entry['case'], **native_coverage(CERT_ROOT / entry['case'] / 'run'))
            index.setdefault('pending_after_feasibility', {})[entry['case']] = entry
        n = sum(v['feasible'] for v in feas.values())
        index['feasibility'] = dict(facts=feas, feasible=n, threshold=2, go=n >= 2)
        write(index_path, index)
        log(dict(event='feasibility', feasible=n))
        if n < 2:
            index['stopped'] = f'feasibility gate: {n}/5 facts natively certifiable in both orders (< 2); no edited service run'
            write(index_path, index)
            return 3
    # repair A1: finish cases left 'native_done' by an earlier run (their native services are reused as is)
    for name, entry in sorted(index['cases'].items()):
        if entry.get('state') == 'native_done':
            entry = run_case(entry, CERT_ROOT, log)
            index['cases'][name] = entry
            log(dict(event='case_done', case=name, state=entry['state'], verdict=entry.get('verdict'), repair='A1'))
            write(index_path, index)
            if entry['state'] == 'service_failed':
                index['stopped'] = f"service failure in {name}: series stopped, no retry"
                write(index_path, index)
                return 1
    index['repairs'] = sorted(set(index.get('repairs', [])) | {'A1: native_done cases resumed'}) if any(
        e.get('state') == 'certified' for e in index['cases'].values()) else index.get('repairs', [])
    while True:
        inv = json.loads(INVENTORY.read_text())
        assert inv['blind_set_opened'] is False
        todo = [c for c in inv['cases'] if c['status'] == 'complete' and case_name(c) not in index['cases']]
        for c in todo:
            pending = index.get('pending_after_feasibility', {}).pop(case_name(c), None)
            entry = pending or prepare_case(c, CERT_ROOT)
            if not pending:
                log(dict(event='prepared', **entry))
            # repair A1 (protocol v2 amendment): a case whose native services ran in the feasibility gate is
            # 'native_done' and must still run its remaining services; v2 run 1 skipped them
            if entry['state'] in ('planned', 'native_done'):
                entry = run_case(entry, CERT_ROOT, log)
            index['cases'][entry['case']] = entry
            write(index_path, index)
            log(dict(event='case_done', case=entry['case'], state=entry['state'], verdict=entry.get('verdict')))
            if entry['state'] == 'service_failed':
                index['stopped'] = f"service failure in {entry['case']}: series stopped, no retry"
                write(index_path, index)
                return 1
        open_cases = [c for c in inv['cases'] if c['status'] not in ('complete', 'failed')]
        if not open_cases and not todo:
            index['inventory_failed'] = [case_name(c) for c in inv['cases'] if c['status'] == 'failed']
            index['complete'] = True
            write(index_path, index)
            return 0
        if time.time() - start > WALL_LIMIT:
            index['stopped'] = 'wall-clock limit reached with solves still pending'
            write(index_path, index)
            return 2
        if not todo:
            time.sleep(60)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rehearse', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--i-have-approval', action='store_true')
    ap.add_argument('--v2', action='store_true')
    a = ap.parse_args()
    if a.v2:
        use_v2()
    if a.rehearse:
        sys.exit(rehearse())
    if a.run and a.i_have_approval:
        sys.exit(run_all())
    sys.exit('refused: use --rehearse, or --run --i-have-approval')
