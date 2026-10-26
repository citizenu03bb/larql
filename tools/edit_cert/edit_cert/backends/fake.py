"""Planted edits for the fake instrument. Tests only."""
from __future__ import annotations

import contextlib

from ..instruments.fake import Override
from .base import Backend, register

STATIC_FORM = dict(answer_time_trigger=False, donor_pass=False, uses_native_proposal=False,
                   substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=True,
                   max_write_layer=8)


@register
class FakeEdit(Backend):
    kind = 'fake'

    def __init__(self, overrides: list[Override], label='', seen_phrasings=None, form=None,
                 persists=True, one_shot=False):
        super().__init__(label, seen_phrasings)
        self.overrides, self.persists, self.one_shot = list(overrides), persists, one_shot
        self._form = dict(STATIC_FORM, **(form or {}))
        if one_shot:
            self._form.update(answer_time_trigger=True, applied_every_forward=False)

    def form(self):
        return dict(self._form)

    def params(self):
        return dict(overrides=[o.to_json() for o in self.overrides], persists=self.persists,
                    one_shot=self.one_shot, form=self._form)

    @contextlib.contextmanager
    def apply(self, inst):
        # the runner marks the copy it reloads; a non-persistent fake loses its state there
        lost = getattr(self, 'loaded_from_saved_copy', False) and not self.persists
        inst.model.overrides = [] if lost else list(self.overrides)
        inst.model.inert_after = 1 if self.one_shot else None
        inst.model.forwards = 0
        try:
            yield
        finally:
            inst.model.overrides, inst.model.inert_after = [], None

    def random_counterpart(self, seed):
        return FakeEdit([], label=f'random({self.label}, seed={seed})', form=self._form)

    @classmethod
    def from_parts(cls, label, seen, params, arrays):
        return cls([Override(**o) for o in params['overrides']], label, seen, params['form'],
                   params['persists'], params['one_shot'])
