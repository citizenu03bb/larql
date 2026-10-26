"""Evidence records: what a live run or a replay adapter hands to the controls.

Two kinds, kept apart on purpose:
- Generation: free greedy generation from an empty chat answer. Only this kind can make a control PASS.
- Boundary: one teacher-forced forward on a resumed (native) prefix. A diagnostic. It can make a control
  FAIL (an observed flip or non-flip is a fact about the model), never PASS.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .spec import Reading

CONDITIONS = ('native', 'edit', 'random', 'donor', 'reloaded')


@dataclass(frozen=True)
class Generation:
    question_id: str
    condition: str
    ids: tuple[int, ...]
    text: str
    eos: bool
    capped: bool
    first_logits_sha256: str | None = None
    trigger_in_prompt: bool | None = None   # trigger-keyed backends: does the prompt hold a trigger token?
    source: str = ''


@dataclass(frozen=True)
class Boundary:
    question_id: str
    condition: str
    native_id: int            # the native value token at this boundary
    target_id: int | None     # the edit's target token (efficacy) or a rejected value (attribute)
    top1: int
    margin: float | None      # logit[target] - logit[native]
    mention_order: str        # which mention the edit acted on: before/after the relation, or 'all'
    partial: bool = False     # the edit applied to a subset of its mentions (a row split): used by C3 only
    variant: str = ''
    source: str = ''


@dataclass
class Evidence:
    instrument: str
    synthetic: bool
    kv_write_deadline: int | None = None    # last block whose output still reaches later positions
    generations: list[Generation] = field(default_factory=list)
    boundaries: list[Boundary] = field(default_factory=list)
    artifact_sha256: dict[str, str] = field(default_factory=dict)   # condition -> backend artifact hash used
    notes: list[str] = field(default_factory=list)

    def generation(self, qid: str, condition: str) -> Generation | None:
        found = [g for g in self.generations if g.question_id == qid and g.condition == condition]
        if len(found) > 1:
            raise ValueError(f'duplicate generation {qid}/{condition}')
        return found[0] if found else None

    def boundaries_for(self, qid: str, condition: str) -> list[Boundary]:
        return [b for b in self.boundaries if b.question_id == qid and b.condition == condition]

    def to_json(self) -> dict:
        return dict(instrument=self.instrument, synthetic=self.synthetic, kv_write_deadline=self.kv_write_deadline,
                    generations=[asdict(g) for g in self.generations],
                    boundaries=[asdict(b) for b in self.boundaries],
                    artifact_sha256=self.artifact_sha256, notes=self.notes)

    @classmethod
    def from_json(cls, raw: dict) -> 'Evidence':
        return cls(raw['instrument'], raw['synthetic'], raw.get('kv_write_deadline'),
                   [Generation(**{**g, 'ids': tuple(g['ids'])}) for g in raw['generations']],
                   [Boundary(**b) for b in raw['boundaries']], raw.get('artifact_sha256', {}), raw.get('notes', []))

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1) + '\n')


def _word(surface: str, text: str) -> bool:
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(surface) + r'(?![A-Za-z0-9])', text) is not None


def contains(reading: Reading, gen: Generation, instrument: str, which: str = 'accept') -> bool:
    """Token ids decide when the spec gives them for this instrument; surfaces otherwise."""
    ids = (reading.accept_ids if which == 'accept' else reading.reject_ids).get(instrument)
    if ids:
        return any(i in gen.ids for i in ids)
    surfaces = reading.accept if which == 'accept' else reading.reject
    return any(_word(s, gen.text) for s in surfaces)


def has_swap(markers: tuple[str, ...], gen: Generation) -> bool:
    return any(m.lower() in gen.text.lower() for m in markers)
