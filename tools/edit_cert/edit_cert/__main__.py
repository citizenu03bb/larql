"""edit_cert command line.

  plan     --run DIR --fact SPEC.json --edit EDIT_DIR --instrument {fake,nf4-e4b,nf4-e2b}   (model-free)
  run      --run DIR --service K [--model-root PATH]       (one service; use run_service.sh for real models)
  verdict  --run DIR                                       (model-free)
  replay   [--root ARCHIVES] [--out DIR]                   (model-free, exit criterion)
  selftest [--out DIR]                                     (model-free, fake instrument)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import runner


def instrument(name: str, model_root: Path | None):
    if name == 'fake':
        from .instruments.fake import FakeInstrument
        return FakeInstrument()
    if name in ('nf4-e4b', 'nf4-e2b'):
        from .instruments.nf4 import Nf4Instrument
        return Nf4Instrument.load(name.split('-')[1].upper(), model_root or Path('/mnt/models/hf-models'))
    raise SystemExit(f'unknown instrument {name}')


def selftest(out: Path) -> dict:
    from .backends.fake import FakeEdit
    from .instruments.fake import FakeInstrument, Override
    spec = Path(__file__).resolve().parents[1] / 'tests' / 'fake-hydrogen-symbol.json'
    edit = FakeEdit([Override('hydrogen', 'symbol', 'He')], label='planted selective edit (synthetic)',
                    seen_phrasings=["Give hydrogen's chemical symbol."])
    edit.save(out / 'edit')
    plan = runner.prepare(out / 'run', spec, out / 'edit', 'fake')
    inst = FakeInstrument()
    for k in range(len(plan['services'])):
        runner.run_service(out / 'run', k, inst)
    return runner.certify(out / 'run')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='edit_cert')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan'); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--fact', type=Path, required=True); p.add_argument('--edit', type=Path, required=True)
    p.add_argument('--instrument', required=True)
    r = sub.add_parser('run'); r.add_argument('--run', type=Path, required=True)
    r.add_argument('--service', type=int, required=True); r.add_argument('--model-root', type=Path)
    v = sub.add_parser('verdict'); v.add_argument('--run', type=Path, required=True)
    rp = sub.add_parser('replay'); rp.add_argument('--root', type=Path); rp.add_argument('--out', type=Path)
    st = sub.add_parser('selftest'); st.add_argument('--out', type=Path)
    a = ap.parse_args(argv)

    if a.cmd == 'plan':
        plan = runner.prepare(a.run, a.fact, a.edit, a.instrument)
        for s in plan['services']:
            print(f"service {s['index']:2}  {s['phase']:6}  {s['forwards']:2} forwards  {len(s['items'])} items")
        print(f"total {plan['total_forwards']} forwards in {len(plan['services'])} services; plan {plan['plan_sha256'][:12]}")
    elif a.cmd == 'run':
        plan = json.loads((a.run / 'plan.json').read_text())
        print(json.dumps(runner.run_service(a.run, a.service, instrument(plan['instrument'], a.model_root))))
    elif a.cmd == 'verdict':
        v = runner.certify(a.run)
        print(v['verdict'], [r['control'] for r in v['reasons']])
    elif a.cmd == 'replay':
        from .replay import replay_all
        from .replay.archives import DEFAULT_ROOT
        s = replay_all(a.root or DEFAULT_ROOT, a.out)
        for cid, rep in s['cases'].items():
            print(f"{cid} {rep['verdict']:9} expected={'yes' if not rep['mismatch'] else rep['mismatch']} "
                  f"reproduced={rep['reproduced'] - len(rep['unreproduced'])}/{rep['reproduced']}")
        print('exit criterion met:', s['exit_criterion_met'])
        return 0 if s['exit_criterion_met'] else 1
    elif a.cmd == 'selftest':
        out = a.out or Path(tempfile.mkdtemp(prefix='edit-cert-selftest-'))
        v = selftest(out)
        print(v['verdict'], [r['control'] for r in v['reasons']], out)
        return 0 if v['verdict'] == 'CERTIFIED' else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
