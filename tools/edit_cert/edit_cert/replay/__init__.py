"""Exit-criterion replay: archived cases R1..R5 against their expected verdicts."""
from __future__ import annotations

import json
from pathlib import Path

from ..verdict import decide, write as write_verdict
from .archives import CASES, DEFAULT_ROOT, EXPECTED


def replay(case_id: str, root: Path = DEFAULT_ROOT) -> tuple[dict, dict]:
    case = CASES[case_id](Path(root))
    v = decide(case.spec, case.form, case.evidence, case.edit_label)
    v['replay'] = dict(case=case.id, title=case.title, reproduced=case.reproduced)
    statuses = {c['id']: c['status'] for c in v['controls']}
    exp = EXPECTED[case_id]
    mismatch = {k: dict(expected=e, got=v['verdict'] if k == 'verdict' else statuses[k])
                for k, e in exp.items() if e != (v['verdict'] if k == 'verdict' else statuses[k])}
    unreproduced = [r for r in case.reproduced if not r['equal']]
    report = dict(case=case_id, title=case.title, verdict=v['verdict'], expected=exp, mismatch=mismatch,
                  reproduced=len(case.reproduced), unreproduced=unreproduced,
                  ok=not mismatch and not unreproduced and bool(case.reproduced))
    return v, report


def replay_all(root: Path = DEFAULT_ROOT, out: Path | None = None) -> dict:
    reports = {}
    for cid in CASES:
        v, rep = replay(cid, root)
        reports[cid] = rep
        if out is not None:
            write_verdict(v, Path(out), stem=f'{cid}-verdict')
    summary = dict(archive_root=str(root), cases=reports, exit_criterion_met=all(r['ok'] for r in reports.values()))
    if out is not None:
        (Path(out) / 'replay-report.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
        (Path(out) / 'replay-report.md').write_text(render(summary))
    return summary


def render(summary: dict) -> str:
    lines = ['# Replay of archived cases', '', f"Exit criterion met: **{summary['exit_criterion_met']}**", '',
             '| Case | Title | Verdict | Matches expected | Numbers reproduced |', '|---|---|---|---|---|']
    for cid, r in summary['cases'].items():
        lines.append(f"| {cid} | {r['title']} | {r['verdict']} | {'yes' if not r['mismatch'] else r['mismatch']} | "
                     f"{r['reproduced'] - len(r['unreproduced'])}/{r['reproduced']} |")
    return '\n'.join(lines) + '\n'
