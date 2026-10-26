"""Model-free plan: every generation and boundary forward, its worst-case cost, packed into services.

Phases run in order, and a service holds items of one phase only:
  native  - native generations of every question;
  edit    - edited generations, random-direction generations, resumed-boundary diagnostics and donor
            ceilings (these need the native answers);
  reload  - edited generations again, in a later service, from the saved copy of the edit (C7).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .spec import FactSpec

SERVICE_CAP = 24
# Caps other than the default need an authority. 400 forwards per service: Pat's Phase B decision Q1
# (docs/phase-b-decision.md §7), NF4 bench only. The fake instrument may use it in tests.
APPROVED_CAPS = {400: dict(authority='phase-b-decision Q1 (Pat, 2026-10-26)', instruments=('nf4-e2b', 'nf4-e4b', 'fake'))}


def check_cap(cap: int, instrument: str) -> dict | None:
    if cap == SERVICE_CAP:
        return None
    rule = APPROVED_CAPS.get(cap)
    if rule is None or instrument not in rule['instruments']:
        raise ValueError(f'service cap {cap} is not approved for instrument {instrument}')
    return dict(cap=cap, authority=rule['authority'])


@dataclass(frozen=True)
class Item:
    id: str
    phase: str
    kind: str          # 'generate' | 'boundary'
    question_id: str
    condition: str     # native | edit | random | donor | reloaded
    cost: int


def items(spec: FactSpec, donor: bool = True) -> list[Item]:
    T = spec.max_new_tokens
    out = [Item(f'gen:{q.id}:native', 'native', 'generate', q.id, 'native', T) for q in spec.questions]
    for q in spec.questions:
        out.append(Item(f'gen:{q.id}:edit', 'edit', 'generate', q.id, 'edit', T))
        if q.control == 'efficacy' and not q.seen:
            out.append(Item(f'gen:{q.id}:random', 'edit', 'generate', q.id, 'random', T))
            out += [Item(f'bnd:{q.id}:{c}', 'edit', 'boundary', q.id, c, 1) for c in ('native', 'edit', 'random')]
            if donor and spec.donor:
                out.append(Item(f'bnd:{q.id}:donor', 'edit', 'boundary', q.id, 'donor', 1))
        if q.control == 'attribute':
            out += [Item(f'bnd:{q.id}:{c}', 'edit', 'boundary', q.id, c, 1) for c in ('native', 'edit')]
    out += [Item(f'gen:{q.id}:reloaded', 'reload', 'generate', q.id, 'reloaded', T)
            for q in spec.by_control('efficacy') if not q.seen]
    return out


def pack(all_items: list[Item], cap: int = SERVICE_CAP) -> list[list[Item]]:
    services: list[list[Item]] = []
    for phase in ('native', 'edit', 'reload'):
        current, used = [], 0
        for it in (i for i in all_items if i.phase == phase):
            if it.cost > cap:
                raise ValueError(f'{it.id} costs {it.cost} forwards, above the service cap {cap}')
            if used + it.cost > cap:
                services.append(current)
                current, used = [], 0
            current.append(it)
            used += it.cost
        if current:
            services.append(current)
    return services


def build(spec: FactSpec, spec_sha256: str, backend_meta: dict, instrument: str, cap: int = SERVICE_CAP) -> dict:
    approval = check_cap(cap, instrument)
    services = pack(items(spec), cap)
    plan = dict(fact_id=spec.fact_id, spec_sha256=spec_sha256, instrument=instrument, cap=cap, cap_approval=approval,
                backend=dict(kind=backend_meta['kind'], label=backend_meta['label'],
                             content_sha256=backend_meta['content_sha256']),
                services=[dict(index=k, phase=s[0].phase, forwards=sum(i.cost for i in s),
                               items=[asdict(i) for i in s]) for k, s in enumerate(services)],
                total_forwards=sum(i.cost for s in services for i in s))
    plan['plan_sha256'] = digest(plan)
    return plan


def digest(plan: dict) -> str:
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load(run_dir: Path) -> dict:
    plan = json.loads((Path(run_dir) / 'plan.json').read_text())
    if digest(plan) != plan['plan_sha256']:
        raise ValueError('plan.json was modified after it was written')
    return plan
