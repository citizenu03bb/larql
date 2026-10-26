"""A tiny deterministic word-level 'model' with a planted fact table. Tests only; never evidence.

A question is parsed for a subject and a relation keyword; the answer follows a fixed template, one word per
token, then EOS. Edits are overrides consulted while answering. Logits are 10 on the next template token and 0
elsewhere, so resumed-boundary margins are -10 (native) or +10 (flipped).
"""
from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

EOS, BOS, SEP = 0, 1, 2
VOCAB = 4096

FACTS = {
    ('hydrogen', 'symbol'): 'H', ('hydrogen', 'number'): '1', ('hydrogen', 'mass'): '1.008',
    ('helium', 'symbol'): 'He', ('helium', 'number'): '2', ('helium', 'mass'): '4.003',
    ('lithium', 'symbol'): 'Li', ('oxygen', 'symbol'): 'O', ('france', 'capital'): 'Paris',
    ('water', 'made'): 'hydrogen and oxygen', ('element', 'number'): 'hydrogen',
}
RELATIONS = ('symbol', 'number', 'mass', 'capital', 'made', 'one')
SUBJECTS = ('hydrogen', 'helium', 'lithium', 'oxygen', 'france', 'water', 'element')


def words(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+(?:'s)?|[?.,!]", text)


class Override:
    """Replace (subject, relation)'s value; optionally only when the subject precedes / follows the relation,
    and optionally rename the subject in every answer (an entity swap)."""

    def __init__(self, subject, relation, value, when='any', rename=None):
        self.subject, self.relation, self.value, self.when, self.rename = subject, relation, value, when, rename

    def to_json(self):
        return dict(subject=self.subject, relation=self.relation, value=self.value, when=self.when, rename=self.rename)


class FakeModel:
    def __init__(self):
        self.overrides: list[Override] = []
        self.inert_after: int | None = None   # one-shot style: overrides stop after this many forwards
        self.forwards = 0

    def answer(self, question_words: list[str]) -> list[str]:
        low = [w.lower().removesuffix("'s") for w in question_words]
        subj = next((w for w in low if w in SUBJECTS), None)
        rel = next((w for w in low if w in RELATIONS), None)
        if subj is None or rel is None or (subj, rel) not in FACTS:
            return ['I', 'do', 'not', 'know', '.']
        order = 'subject_before_relation' if low.index(subj) < low.index(rel) else 'subject_after_relation'
        value, name = FACTS[(subj, rel)], subj
        active = self.inert_after is None or self.forwards <= self.inert_after
        for o in (self.overrides if active else []):
            if o.subject == subj and o.relation == rel and o.when in ('any', order):
                value = o.value
            if o.rename and o.subject == subj:
                name = o.rename
            if o.rename:
                value = value.replace(o.subject, o.rename)
        if subj == 'water':
            return ['Water', 'is', 'made', 'of'] + value.split() + ['.']
        if subj == 'element':
            return ['The', 'element', 'is'] + value.split() + ['.']
        return [name.capitalize() + "'s", rel, 'is', value, '.']


class FakeInstrument:
    synthetic = True

    def __init__(self, name='fake', kv_write_deadline=22):
        self.name = name
        self.kv_write_deadline = kv_write_deadline
        self.model = FakeModel()
        self.layers = None
        self.eos_ids = frozenset({EOS})
        self.current_ids: list[int] = []
        self._words: dict[int, str] = {EOS: '<eos>', BOS: '<bos>', SEP: '<sep>'}

    def token(self, word: str) -> int:
        i = 3 + zlib.crc32(word.encode()) % (VOCAB - 3)
        seen = self._words.setdefault(i, word)
        assert seen == word, f'fake vocabulary collision: {seen!r} / {word!r}'
        return i

    def token_ids(self, surface: str) -> list[int]:
        return [self.token(w) for w in words(surface)]

    def chat_prompt(self, question: str) -> list[int]:
        return [BOS] + [self.token(w) for w in words(question)] + [SEP]

    def decode(self, ids) -> str:
        return ' '.join(self._words[i] for i in ids if i not in (EOS, BOS, SEP))

    def forward(self, ids: list[int]) -> np.ndarray:
        self.current_ids = list(ids)
        self.model.forwards += 1
        sep = ids.index(SEP)
        question = [self._words[i] for i in ids[1:sep]]
        so_far = list(ids[sep + 1:])
        template = [self.token(w) for w in self.model.answer(question)] + [EOS]
        # Position-indexed, so a resumed prefix (e.g. a donor prompt with the native answer prefix) still reads on.
        nxt = template[len(so_far)] if len(so_far) < len(template) else EOS
        logits = np.zeros(VOCAB, dtype=np.float32)
        logits[nxt] = 10.0
        return logits

    def fingerprint(self) -> dict:
        return dict(instrument=self.name, synthetic=True,
                    facts_sha256=hashlib.sha256(repr(sorted(FACTS.items())).encode()).hexdigest())
