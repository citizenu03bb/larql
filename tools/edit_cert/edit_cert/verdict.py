"""Verdict rule and rendering. CERTIFIED iff every mandatory control is PASS (design §3, Q1)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import controls as C
from .evidence import Evidence
from .spec import FactSpec

CERTIFIED, REFUSED = 'CERTIFIED', 'REFUSED'


def decide(spec: FactSpec, form: dict, ev: Evidence, edit_label: str = '') -> dict:
    results = C.run_all(spec, form, ev)
    reasons = [dict(control=r.id, name=r.name, status=r.status, basis=r.basis, detail=r.detail)
               for r in results if r.mandatory and r.status != C.PASS]
    verdict = CERTIFIED if not reasons else REFUSED
    quals = C.qualify(spec, ev)
    return dict(
        verdict=verdict, reasons=reasons, synthetic=ev.synthetic,
        scope=dict(fact=spec.fact_id, edit=edit_label, instrument=ev.instrument, decoding='greedy, one trial',
                   questions=[q.id for q in spec.questions],
                   claim='this edit, this fact, this question set, this instrument; not a generalisation'),
        controls=[dict(id=r.id, name=r.name, mandatory=r.mandatory, status=r.status, basis=r.basis,
                       detail=r.detail, data=r.data, outcomes=[asdict(o) for o in r.outcomes]) for r in results],
        qualification={k: dict(level=v[0], why=v[1]) for k, v in quals.items()},
        notes=ev.notes)


def render_md(v: dict) -> str:
    head = ('**SYNTHETIC: test fixture, not evidence.**\n\n' if v['synthetic'] else '')
    lines = [f"# Edit certification: {v['verdict']}", '', head + f"- Fact: `{v['scope']['fact']}`",
             f"- Edit: {v['scope']['edit'] or '(unnamed)'}", f"- Instrument: `{v['scope']['instrument']}`",
             f"- Decoding: {v['scope']['decoding']}", f"- Scope: {v['scope']['claim']}", '']
    if v['reasons']:
        lines += ['## Reasons', '']
        lines += [f"- **{r['control']}** {r['name']}: {r['status']}"
                  + (f" ({r['basis']})" if r['basis'] not in ('none', 'form', 'generation') else '')
                  + f" — {r['detail']}" for r in v['reasons']]
        lines.append('')
    lines += ['## Controls', '', '| Control | Mandatory | Status | Basis | Detail |', '|---|---|---|---|---|']
    for c in v['controls']:
        detail = c['detail'].replace('|', '\\|')
        lines.append(f"| {c['id']} {c['name']} | {'yes' if c['mandatory'] else 'no'} | {c['status']} | {c['basis']} | {detail} |")
    lines += ['', 'A resumed-boundary diagnostic can refuse an edit but never certify one.', '']
    if v['notes']:
        lines += ['## Notes', ''] + [f'- {n}' for n in v['notes']] + ['']
    return '\n'.join(lines)


def write(v: dict, out: Path, stem: str = 'verdict') -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{stem}.json').write_text(json.dumps(v, indent=1, sort_keys=True) + '\n')
    (out / f'{stem}.md').write_text(render_md(v))
