"""P1-1 native qualification on GPT-2 XL (Claude Code), under docs/phase-b-p1-positive-control-protocol.md.

  python3 scripts/p1_qualify.py [--v2] --rehearse              # model-free: plans in a scratch root, wrapper refusal
  python3 scripts/p1_qualify.py [--v2] --run --i-have-approval # native services only, then the feasibility table

--v2 (docs/phase-b-p1-v2-protocol.md): the rewritten templates (facts/p1-gpt2xl-v2), a new root and protocol.

Each fact is planned by the harness with a placeholder rank-1 edit of zeros at block 17 (native services never
apply the edit), and only its native-phase services run, through run_service.sh. Feasibility uses the harness's
own `qualify`, as in the B2 v2 gate: at least 2 unseen efficacy phrasings covering both subject orders, and at
least one attribute in each order.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from edit_cert import controls, plan as planner, runner, spec as specs  # noqa: E402
from edit_cert.backends.lowrank import LowRank  # noqa: E402
from edit_cert.evidence import Evidence, Generation  # noqa: E402

PROTOCOL = 'docs/phase-b-p1-positive-control-protocol.md'
SPECS = HERE / 'facts' / 'p1-gpt2xl'
ROOT = Path.home() / 'Projects/larql-research/phase-b-p1-20261027/qualification'
INSTRUMENT = 'hf-gpt2xl'
BEFORE, AFTER = 'subject_before_relation', 'subject_after_relation'
THRESHOLD = 5


def use_v2():
    global PROTOCOL, SPECS, ROOT
    PROTOCOL = 'docs/phase-b-p1-v2-protocol.md'
    SPECS = HERE / 'facts' / 'p1-gpt2xl-v2'
    ROOT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027/qualification'


def placeholder(dirpath: Path, seen: str) -> None:
    edit = LowRank([dict(module='transformer.h.17.mlp.c_proj', layer=17, scale=1.0)],
                   dict(U0=np.zeros((1600, 1), np.float32), V0=np.zeros((6400, 1), np.float32)),
                   label='P1-1 placeholder (zeros; native services only)', seen_phrasings=[seen])
    edit.save(dirpath)


def prepare(root: Path) -> list[dict]:
    out = []
    for sp in sorted(SPECS.glob('*.json')):
        s = specs.load(sp)
        d = root / s.fact_id
        d.mkdir(parents=True, exist_ok=False)
        placeholder(d / 'edit', next(q.text for q in s.questions if q.seen))
        plan = runner.prepare(d / 'run', sp, d / 'edit', INSTRUMENT, cap=400)
        native = [k for k, svc in enumerate(plan['services']) if svc['phase'] == 'native']
        assert native == list(range(len(native))), 'native services must come first'
        out.append(dict(fact_id=s.fact_id, native_services=native,
                        forwards=[plan['services'][k]['forwards'] for k in native], plan_sha256=plan['plan_sha256']))
    return out


def coverage(run: Path) -> dict:
    sp = specs.load(run / 'spec.json')
    ev = Evidence(INSTRUMENT, False)
    for st in sorted((run / 'services').glob('*/records.json')):
        r = json.loads(st.read_text())
        ev.generations += [Generation(**{**g, 'ids': tuple(g['ids'])}) for g in r['generations']]
    q = controls.qualify(sp, ev)

    def orders(ctl):
        return sorted({x.order for x in sp.questions if x.control == ctl and not x.seen and q[x.id][0] == 'generation'})
    eff = [x.id for x in sp.questions if x.control == 'efficacy' and not x.seen and q[x.id][0] == 'generation']
    cov = dict(efficacy_orders=orders('efficacy'), attribute_orders=orders('attribute'), efficacy_count=len(eff),
               qualified={k: v[0] for k, v in q.items()}, excluded={k: v[1] for k, v in q.items() if v[0] == 'excluded'},
               answers={g.question_id: g.text for g in ev.generations})
    cov['feasible'] = ({BEFORE, AFTER} <= set(cov['efficacy_orders']) and len(eff) >= sp.min_efficacy
                       and {BEFORE, AFTER} <= set(cov['attribute_orders']))
    return cov


def rehearse() -> int:
    tmp = Path(tempfile.mkdtemp(prefix='p1-qualify-rehearsal-'))
    entries = prepare(tmp)
    total = sum(sum(e['forwards']) for e in entries)
    assert all(f <= 400 for e in entries for f in e['forwards']) and total <= 3120, total
    r = subprocess.run(['bash', str(HERE / 'run_service.sh'), '--protocol', PROTOCOL, '--run',
                        str(tmp / entries[0]['fact_id'] / 'run'), '--service', '0'], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 3 and 'no --i-have-approval' in r.stderr, r
    print(json.dumps(dict(rehearsal='ok', facts=len(entries), native_forwards_bound=total,
                          services=[e['forwards'] for e in entries], scratch=str(tmp))))
    return 0


def run_all() -> int:
    ROOT.mkdir(parents=True, exist_ok=False)
    entries = prepare(ROOT)
    index = dict(author='Claude Code', protocol=PROTOCOL, stage='P1-1 v2' if 'v2' in PROTOCOL else 'P1-1', facts={})
    log = open(ROOT / 'driver-log.jsonl', 'a')
    for e in entries:
        run = ROOT / e['fact_id'] / 'run'
        for k in e['native_services']:
            t0 = time.time()
            p = subprocess.run(['bash', str(HERE / 'run_service.sh'), '--protocol', PROTOCOL, '--run', str(run),
                                '--service', str(k), '--i-have-approval'], cwd=REPO, capture_output=True, text=True)
            log.write(json.dumps(dict(fact=e['fact_id'], service=k, rc=p.returncode, seconds=round(time.time() - t0, 1),
                                      stderr_tail=p.stderr[-800:])) + '\n')
            log.flush()
            if p.returncode != 0 or not (run / 'services' / str(k) / 'status.json').exists():
                index['stopped'] = f"service failure in {e['fact_id']} service {k}: stage stopped, no retry"
                (ROOT / 'index.json').write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
                return 1
        index['facts'][e['fact_id']] = dict(e, **coverage(run))
    n = sum(v['feasible'] for v in index['facts'].values())
    index.update(feasible=n, threshold=THRESHOLD, go=n >= THRESHOLD)
    (ROOT / 'index.json').write_text(json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
    print(json.dumps(dict(feasible=n, go=n >= THRESHOLD)))
    return 0


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
