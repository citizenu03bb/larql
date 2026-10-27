"""Model-free check of the P1-5 bench (Claude Code): plans against the committed index, forward counts against the
plans, and each verdict recomputed from the saved records with the harness's own `decide`.

  python3 scripts/p1_verify_certification.py OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from edit_cert import backends, plan as planner, runner, spec as specs  # noqa: E402
from edit_cert.verdict import decide  # noqa: E402

CERT = Path.home() / 'Projects/larql-research/phase-b-p1-v2-20261027/certification'


def main(out: Path) -> int:
    idx = json.loads((REPO / 'docs/research-data/phase-b-p1-5-plans-claude.json').read_text())
    saved = json.loads((CERT / 'index.json').read_text())
    facts = {}
    for fact, p in sorted(idx['facts'].items()):
        run = CERT / fact / 'run'
        plan = planner.load(run)
        counts = [json.loads((run / 'services' / str(k) / 'status.json').read_text())['forwards']
                  for k in range(len(p['services']))]
        v = decide(specs.load(run / 'spec.json'), backends.load(run / 'edit-source').form(), runner.collect(run),
                   backends.load(run / 'edit-source').label)
        old = json.loads((run / 'verdict.json').read_text())
        same = (v['verdict'] == old['verdict']
                and [(c['id'], c['status']) for c in v['controls']] == [(c['id'], c['status']) for c in old['controls']])
        facts[fact] = dict(plan_matches_commit=plan['plan_sha256'] == p['plan_sha256'],
                           edit_matches_handoff=plan['backend']['content_sha256'] == p['edit_content_sha256'],
                           counts=counts, within_plan=all(c <= s for c, s in zip(counts, p['services'])),
                           recomputed_verdict=v['verdict'], verdict_reproduced=same,
                           index_verdict=saved['facts'][fact]['verdict'])
    ok = all(f['plan_matches_commit'] and f['edit_matches_handoff'] and f['within_plan'] and f['verdict_reproduced']
             and f['recomputed_verdict'] == f['index_verdict'] for f in facts.values())
    res = dict(author='Claude Code (driver and verifier by the same agent; Codex cross-checks)', ok=ok,
               bench_forwards=sum(sum(f['counts']) for f in facts.values()), facts=facts)
    out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(ok=ok, bench_forwards=res['bench_forwards'],
                          verdicts={k: f['recomputed_verdict'] for k, f in facts.items()})))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
