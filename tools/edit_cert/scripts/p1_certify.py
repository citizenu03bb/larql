"""P1-5 certification of Codex's five A3-agreeing edits on GPT-2 XL (Claude Code), under docs/phase-b-p1-amendment-a3.md.

  python3 scripts/p1_certify.py --rehearse                  # model-free: plans in a scratch root, caps, wrapper refusal
  python3 scripts/p1_certify.py --plan                      # writes the real plans and their index (commit it first)
  python3 scripts/p1_certify.py --run --i-have-approval     # services, one at a time, then the verdicts

Inputs come from Codex's handoff (docs/research-data/phase-b-p1-a3-certification-handoff-codex.json on
research/phase-b-rome, 7a3824a5): each unchanged Codex edit (backend.json, U0, V0) and its v2 spec. Each fact's plan
must equal its A3 bench cap exactly, in services of at most 400. Before each service the plan is checked against
the committed plan index. Any failed service stops P1-5, with no retry. Controls and readings are the harness's,
unchanged, with the C7 rule of design §14 and the negation rule of §15.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from edit_cert import plan as planner, runner  # noqa: E402

PROTOCOL = 'docs/phase-b-p1-amendment-a3.md'
ROOT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027/certification'
HANDOFF = Path.home() / 'Projects/larql.worktree/phase-b-rome/docs/research-data/phase-b-p1-a3-certification-handoff-codex.json'
PLAN_INDEX = REPO / 'docs/research-data/phase-b-p1-5-plans-claude.json'
SPECS = HERE / 'facts' / 'p1-gpt2xl-v2'
INSTRUMENT = 'hf-gpt2xl'
MINIMAL = ('backend.json', 'U0.npy', 'V0.npy')


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root: Path) -> dict:
    hand = json.loads(HANDOFF.read_text())
    out = {}
    for fact, c in sorted(hand['cases'].items()):
        spec = SPECS / f'{fact}.json'
        assert sha(spec) == sha(Path(c['spec_path'])) == c['spec_sha256'], f'{fact}: spec differs from the handoff'
        d = root / fact
        (d / 'edit-min').mkdir(parents=True, exist_ok=False)
        for f in MINIMAL:
            shutil.copyfile(Path(c['edit_path']) / f, d / 'edit-min' / f)
        plan = runner.prepare(d / 'run', spec, d / 'edit-min', INSTRUMENT, cap=400)
        assert plan['backend']['content_sha256'] == c['edit_content_sha256'], f'{fact}: edit differs from the handoff'
        svc = [s['forwards'] for s in plan['services']]
        assert all(f <= 400 for f in svc) and sum(svc) == c['bench_forward_cap'], (fact, svc, c['bench_forward_cap'])
        out[fact] = dict(plan_sha256=plan['plan_sha256'], services=svc, phases=[s['phase'] for s in plan['services']],
                         forwards=sum(svc), edit_content_sha256=c['edit_content_sha256'], spec_sha256=c['spec_sha256'])
    return out


def rehearse() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='p1-certify-rehearsal-'))
    plans = prepare(tmp)
    total = sum(p['forwards'] for p in plans.values())
    assert total == 4064, total
    fact = next(iter(plans))
    r = subprocess.run(['bash', str(HERE / 'run_service.sh'), '--protocol', PROTOCOL, '--run', str(tmp / fact / 'run'),
                        '--service', '0'], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 3 and 'no --i-have-approval' in r.stderr, r
    from edit_cert.__main__ import main
    assert main(['selftest', '--out', str(tmp / 'selftest')]) == 0
    print(json.dumps(dict(rehearsal='ok', bench_forwards=total, services={k: v['services'] for k, v in plans.items()},
                          scratch=str(tmp))))
    return 0


def make_plans() -> int:
    ROOT.mkdir(parents=True, exist_ok=False)
    plans = prepare(ROOT)
    idx = dict(author='Claude Code', protocol=PROTOCOL, stage='P1-5 bench', handoff_sha256=sha(HANDOFF),
               bench_forwards=sum(p['forwards'] for p in plans.values()), facts=plans)
    PLAN_INDEX.write_text(json.dumps(idx, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(planned=len(plans), bench_forwards=idx['bench_forwards'])))
    return 0


def run_all() -> int:
    committed = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(PLAN_INDEX)], cwd=REPO).returncode == 0 and \
        subprocess.run(['git', 'ls-files', '--error-unmatch', str(PLAN_INDEX)], cwd=REPO, capture_output=True).returncode == 0
    assert committed, 'refused: the plan index is not committed or differs from HEAD'
    idx = json.loads(PLAN_INDEX.read_text())
    index = dict(author='Claude Code', protocol=PROTOCOL, stage='P1-5 bench', facts={})
    log = open(ROOT / 'driver-log.jsonl', 'a')
    for fact, p in sorted(idx['facts'].items()):
        run = ROOT / fact / 'run'
        assert planner.load(run)['plan_sha256'] == p['plan_sha256'], f'{fact}: plan changed after commit'
        for k in range(len(p['services'])):
            t0 = time.time()
            r = subprocess.run(['bash', str(HERE / 'run_service.sh'), '--protocol', PROTOCOL, '--run', str(run),
                                '--service', str(k), '--i-have-approval'], cwd=REPO, capture_output=True, text=True)
            log.write(json.dumps(dict(fact=fact, service=k, rc=r.returncode, seconds=round(time.time() - t0, 1),
                                      time=time.strftime('%Y-%m-%dT%H:%M:%S'), stderr_tail=r.stderr[-800:])) + '\n')
            log.flush()
            st = run / 'services' / str(k) / 'status.json'
            if r.returncode != 0 or not st.exists() or json.loads(st.read_text())['forwards'] != p['services'][k]:
                index['stopped'] = f'service failure or count mismatch in {fact} service {k}: P1-5 stopped, no retry'
                (ROOT / 'index.json').write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
                return 1
        v = runner.certify(run)
        st = {c['id']: c['status'] for c in v['controls']}
        index['facts'][fact] = dict(verdict=v['verdict'], statuses=st, reasons=[r['control'] for r in v['reasons']],
                                    forwards=p['forwards'], verdict_sha256=sha(run / 'verdict.json'),
                                    evidence_sha256=sha(run / 'evidence.json'))
        (ROOT / 'index.json').write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
    index['certified'] = sum(f['verdict'] == 'CERTIFIED' for f in index['facts'].values())
    (ROOT / 'index.json').write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(certified=index['certified'], verdicts={k: v['verdict'] for k, v in index['facts'].items()})))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rehearse', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--i-have-approval', action='store_true')
    a = ap.parse_args()
    if a.rehearse:
        sys.exit(rehearse())
    if a.plan:
        sys.exit(make_plans())
    if a.run and a.i_have_approval:
        sys.exit(run_all())
    sys.exit('refused: use --rehearse, --plan, or --run --i-have-approval')
