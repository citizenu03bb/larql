"""Fact spec: the target fact, its question set and the readings fixed before any run."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONTROLS = ('efficacy', 'attribute', 'answer_only', 'off_target', 'propagation')
ORDERS = ('subject_before_relation', 'subject_after_relation', 'subject_in_answer_only', 'none')


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Reading:
    """What counts as a value in an answer: surfaces (whole-word text match) and/or token ids per instrument."""
    accept: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()
    accept_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reject_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: dict | None) -> 'Reading':
        raw = raw or {}
        return cls(tuple(raw.get('accept', ())), tuple(raw.get('reject', ())),
                   {k: tuple(v) for k, v in raw.get('accept_ids', {}).items()},
                   {k: tuple(v) for k, v in raw.get('reject_ids', {}).items()})


@dataclass(frozen=True)
class Question:
    id: str
    control: str
    order: str
    text: str
    seen: bool = False
    expect: Reading = Reading()          # attribute / propagation: the value expected after the edit


@dataclass(frozen=True)
class FactSpec:
    fact_id: str
    subject: tuple[str, ...]
    relation: str
    old: Reading
    new: Reading
    swap_markers: tuple[str, ...]
    questions: tuple[Question, ...]
    donor: str | None = None
    min_efficacy: int = 2
    max_new_tokens: int = 12
    random_seed: int = 20261027

    def by_control(self, control: str) -> list[Question]:
        return [q for q in self.questions if q.control == control]

    def question(self, qid: str) -> Question:
        return next(q for q in self.questions if q.id == qid)

    def check_provenance(self, seen_phrasings: list[str]) -> None:
        """Refuse an efficacy question marked unseen whose text was used to build or calibrate the edit."""
        used = {normalise(t) for t in seen_phrasings}
        for q in self.by_control('efficacy'):
            if not q.seen and normalise(q.text) in used:
                raise SpecError(f'{q.id} is marked unseen but its text is in the backend provenance')


def normalise(text: str) -> str:
    return ' '.join(text.lower().split())


def parse(raw: dict) -> FactSpec:
    for key in ('fact_id', 'subject', 'relation', 'old', 'new', 'questions'):
        if key not in raw:
            raise SpecError(f'missing {key}')
    questions = []
    for q in raw['questions']:
        if q.get('control') not in CONTROLS:
            raise SpecError(f"{q.get('id')}: unknown control {q.get('control')!r}")
        if q.get('order') not in ORDERS:
            raise SpecError(f"{q.get('id')}: unknown order {q.get('order')!r}")
        if q['control'] in ('attribute', 'propagation') and not (q.get('expect', {}).get('accept')
                                                                 or q.get('expect', {}).get('accept_ids')):
            raise SpecError(f"{q['id']}: {q['control']} question needs expect.accept")
        questions.append(Question(q['id'], q['control'], q['order'], q['text'], bool(q.get('seen', False)),
                                  Reading.parse(q.get('expect'))))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise SpecError('duplicate question id')
    texts = [normalise(q.text) for q in questions]
    if len(texts) != len(set(texts)):
        raise SpecError('duplicate question text')
    subject = raw['subject']['surface'] if isinstance(raw['subject'], dict) else raw['subject']
    spec = FactSpec(raw['fact_id'], tuple(subject), raw['relation'], Reading.parse(raw['old']),
                    Reading.parse(raw['new']), tuple(raw.get('swap_markers', ())), tuple(questions),
                    raw.get('donor'), int(raw.get('min_efficacy', 2)), int(raw.get('max_new_tokens', 12)),
                    int(raw.get('random_seed', 20261027)))
    if not (spec.old.accept or spec.old.accept_ids) or not (spec.new.accept or spec.new.accept_ids):
        raise SpecError('old and new need accept surfaces or ids')
    return spec


def load(path: Path) -> FactSpec:
    return parse(json.loads(Path(path).read_text()))
