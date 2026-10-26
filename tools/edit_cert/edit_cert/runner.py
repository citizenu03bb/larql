"""Prepare a run, execute one service, collect evidence, decide.

A run directory holds: spec.json, edit-source/ (the edit as given), edit-saved/ (the harness's own save of it,
loaded only by reload services), plan.json, services/<k>/{records.json,status.json}, verdict.{json,md}.
Each service is meant to be its own process (run_service.sh); in tests they run in-process, but the reload
phase still goes through save -> load.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import backends, plan as planner, spec as specs
from .evidence import Boundary, Evidence, Generation
from .verdict import decide, write as write_verdict


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Budget:
    def __init__(self, inst, cap):
        self.inst, self.cap, self.count = inst, cap, 0

    def forward(self, ids):
        if self.count >= self.cap:
            raise RuntimeError(f'service forward cap {self.cap} reached')
        self.count += 1
        return self.inst.forward(ids)


def prepare(run_dir: Path, spec_path: Path, edit_dir: Path, instrument: str, cap: int = planner.SERVICE_CAP) -> dict:
    planner.check_cap(cap, instrument)   # refuse before writing anything
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(spec_path, run_dir / 'spec.json')
    spec = specs.load(run_dir / 'spec.json')
    shutil.copytree(edit_dir, run_dir / 'edit-source')
    edit = backends.load(run_dir / 'edit-source')
    spec.check_provenance(edit.seen_phrasings)
    meta = edit.save(run_dir / 'edit-saved')
    plan = planner.build(spec, sha_file(run_dir / 'spec.json'), meta, instrument, cap)
    (run_dir / 'plan.json').write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n')
    return plan


def _records(run_dir: Path, upto: int | None = None) -> list[dict]:
    out = []
    for status_path in sorted((Path(run_dir) / 'services').glob('*/status.json'), key=lambda p: int(p.parent.name)):
        k = int(status_path.parent.name)
        if upto is not None and k >= upto:
            continue
        status = json.loads(status_path.read_text())
        rec_path = status_path.parent / 'records.json'
        if status.get('status') != 'complete' or sha_file(rec_path) != status['records_sha256']:
            raise ValueError(f'service {k} incomplete or its records were modified')
        out.append(json.loads(rec_path.read_text()))
    return out


def _value_ids(reading, inst, which='accept') -> list[int]:
    ids = (reading.accept_ids if which == 'accept' else reading.reject_ids).get(inst.name)
    if ids:
        return list(ids)
    surfaces = reading.accept if which == 'accept' else reading.reject
    singles = [inst.token_ids(s) for s in surfaces]
    return [t[0] for t in singles if len(t) == 1]


def donor_text(text: str, subject: tuple[str, ...], donor: str) -> str:
    for s in sorted(subject, key=len, reverse=True):
        text = re.sub(re.escape(s), lambda m: donor.capitalize() if m.group(0)[0].isupper() else donor, text)
    return text


def run_service(run_dir: Path, k: int, inst) -> dict:
    run_dir = Path(run_dir)
    plan = planner.load(run_dir)
    svc = plan['services'][k]
    spec = specs.load(run_dir / 'spec.json')
    assert sha_file(run_dir / 'spec.json') == plan['spec_sha256'], 'spec changed after planning'
    assert inst.name == plan['instrument'], (inst.name, plan['instrument'])
    earlier = _records(run_dir, upto=k)
    if len(earlier) != k:
        raise ValueError(f'service {k} needs services 0..{k - 1} complete first')
    out = run_dir / 'services' / str(k)
    out.mkdir(parents=True, exist_ok=False)

    source = 'edit-saved' if svc['phase'] == 'reload' else 'edit-source'
    edit = backends.load(run_dir / source)
    edit.loaded_from_saved_copy = source == 'edit-saved'
    if source == 'edit-source':
        assert edit.content_sha256() == plan['backend']['content_sha256'], 'edit changed after planning'
    random = edit.random_counterpart(spec.random_seed)
    trigger = set(edit.form().get('trigger_token_ids') or [])
    natives = {g['question_id']: g for r in earlier for g in r['generations'] if g['condition'] == 'native'}
    budget = Budget(inst, plan['cap'])
    gens, bnds, notes = [], [], []

    def context(condition):
        return {'edit': edit, 'reloaded': edit, 'random': random}.get(condition)

    for it in svc['items']:
        q = spec.question(it['question_id'])
        cond = it['condition']
        backend = context(cond)
        if it['kind'] == 'generate':
            prompt = inst.chat_prompt(q.text)
            ids, first_sha = [], None
            with (backend.apply(inst) if backend else contextlib.nullcontext()):
                for step in range(spec.max_new_tokens):
                    logits = budget.forward(prompt + ids)
                    if step == 0:
                        first_sha = hashlib.sha256(np.ascontiguousarray(logits, dtype=np.float32).tobytes()).hexdigest()
                    ids.append(int(np.argmax(logits)))
                    if ids[-1] in inst.eos_ids:
                        break
            eos = bool(ids) and ids[-1] in inst.eos_ids
            gens.append(Generation(q.id, cond, tuple(ids), inst.decode(ids), eos, not eos, first_sha,
                                   (any(t in trigger for t in prompt) if trigger else None),
                                   source=f'service {k}'))
            continue
        native = natives.get(q.id)
        reading = spec.old if q.control == 'efficacy' else q.expect
        olds = _value_ids(reading, inst)
        pos = next((i for i, t in enumerate(native['ids']) if t in olds), None) if native else None
        if pos is None:
            notes.append(f'{it["id"]}: no native value token to resume at; boundary skipped')
            continue
        native_id = native['ids'][pos]
        targets = _value_ids(spec.new, inst) if q.control == 'efficacy' else _value_ids(q.expect, inst, 'reject')
        target = (targets[olds.index(native_id)] if len(targets) == len(olds) else targets[0]) if targets else None
        text = donor_text(q.text, spec.subject, spec.donor) if cond == 'donor' else q.text
        prefix = inst.chat_prompt(text) + list(native['ids'][:pos])
        with (backend.apply(inst) if backend else contextlib.nullcontext()):
            logits = budget.forward(prefix)
        margin = float(logits[target]) - float(logits[native_id]) if target is not None else None
        bnds.append(Boundary(q.id, cond, int(native_id), target, int(np.argmax(logits)), margin, 'all',
                             source=f'service {k}'))

    records = dict(service=k, phase=svc['phase'], plan_sha256=plan['plan_sha256'], instrument=inst.name,
                   synthetic=inst.synthetic, kv_write_deadline=inst.kv_write_deadline,
                   fingerprint=inst.fingerprint(), forwards=budget.count,
                   content_sha256={('reloaded' if source == 'edit-saved' else 'edit'): edit.content_sha256()},
                   generations=[asdict(g) for g in gens], boundaries=[asdict(b) for b in bnds], notes=notes)
    (out / 'records.json').write_text(json.dumps(records, indent=1, sort_keys=True) + '\n')
    status = dict(status='complete', forwards=budget.count, records_sha256=sha_file(out / 'records.json'))
    (out / 'status.json').write_text(json.dumps(status, indent=1) + '\n')
    return status


def collect(run_dir: Path) -> Evidence:
    recs = _records(run_dir)
    plan = planner.load(run_dir)
    if len(recs) != len(plan['services']):
        raise ValueError(f'{len(recs)} of {len(plan["services"])} services complete')
    first = recs[0]
    if len({json.dumps(r['fingerprint'], sort_keys=True) for r in recs}) != 1:
        raise ValueError('instrument fingerprint differs between services')
    ev = Evidence(first['instrument'], first['synthetic'], first['kv_write_deadline'])
    for r in recs:
        ev.generations += [Generation(**{**g, 'ids': tuple(g['ids'])}) for g in r['generations']]
        ev.boundaries += [Boundary(**b) for b in r['boundaries']]
        ev.notes += r['notes']
        for cond, sha in r['content_sha256'].items():
            if ev.artifact_sha256.setdefault(cond, sha) != sha:
                raise ValueError(f'{cond} edit content differs between services')
    return ev


def certify(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    spec = specs.load(run_dir / 'spec.json')
    edit = backends.load(run_dir / 'edit-source')   # the form comes from the loaded class, not the JSON
    ev = collect(run_dir)
    ev.write(run_dir / 'evidence.json')
    v = decide(spec, edit.form(), ev, edit.label)
    v['plan_sha256'] = planner.load(run_dir)['plan_sha256']
    write_verdict(v, run_dir)
    return v
