"""Activation-delta backends on a decoder block's output (torch imported lazily)."""
from __future__ import annotations

import contextlib

import numpy as np

from .base import Backend, register, same_norm_gaussian


@register
class ActivationStatic(Backend):
    """Add gain * vector to the output of block `layer` on every row whose input token is in `trigger_ids`,
    at every forward (Phase 3 form). Stateless: no donor pass, no answer-time decision."""
    kind = 'activation_static'

    def __init__(self, layer: int, vector: np.ndarray, gain: float, trigger_ids, label='', seen_phrasings=None):
        super().__init__(label, seen_phrasings)
        self.layer, self.vector, self.gain = int(layer), np.asarray(vector, dtype=np.float32), float(gain)
        self.trigger_ids = tuple(int(t) for t in trigger_ids)

    def form(self):
        return dict(answer_time_trigger=False, donor_pass=False, uses_native_proposal=False,
                    substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=True,
                    max_write_layer=self.layer, trigger_token_ids=list(self.trigger_ids))

    def params(self):
        return dict(layer=self.layer, gain=self.gain, trigger_ids=list(self.trigger_ids))

    def arrays(self):
        return dict(vector=self.vector)

    @contextlib.contextmanager
    def apply(self, inst):
        import torch

        def post(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rows = [i for i, t in enumerate(inst.current_ids) if t in self.trigger_ids]
            if not rows:
                return output
            assert hidden.shape[1] == len(inst.current_ids), 'hook sees a different sequence than the instrument'
            patched = hidden.clone()
            add = torch.as_tensor(self.gain * self.vector, dtype=hidden.dtype, device=hidden.device)
            patched[0, rows] += add
            return (patched,) + tuple(output[1:]) if isinstance(output, tuple) else patched

        handle = inst.layers[self.layer].register_forward_hook(post)
        try:
            yield
        finally:
            handle.remove()

    def random_counterpart(self, seed):
        return ActivationStatic(self.layer, same_norm_gaussian(self.gain * self.vector, seed), 1.0,
                                self.trigger_ids, label=f'random({self.label}, seed={seed})')

    @classmethod
    def from_parts(cls, label, seen, params, arrays):
        return cls(params['layer'], arrays['vector'], params['gain'], params['trigger_ids'], label, seen)


@register
class ActivationOneShot(Backend):
    """The archived form: added once at the native proposal of the old value, then released. It needs the
    native proposal at answer time, so it is not a persistent edit (C0). Replay only in v1."""
    kind = 'activation_oneshot'

    def __init__(self, layer: int, vector: np.ndarray, rows, label='', seen_phrasings=None):
        super().__init__(label, seen_phrasings)
        self.layer, self.vector, self.rows = int(layer), np.asarray(vector, dtype=np.float32), list(rows)

    def form(self):
        return dict(answer_time_trigger=True, donor_pass=False, uses_native_proposal=True,
                    substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=False,
                    max_write_layer=self.layer)

    def params(self):
        return dict(layer=self.layer, rows=self.rows)

    def arrays(self):
        return dict(vector=self.vector)

    def apply(self, inst):
        raise NotImplementedError('activation_oneshot is replay-only in v1: it needs the native proposal')

    def random_counterpart(self, seed):
        return ActivationOneShot(self.layer, same_norm_gaussian(self.vector, seed), self.rows,
                                 label=f'random({self.label}, seed={seed})')

    @classmethod
    def from_parts(cls, label, seen, params, arrays):
        return cls(params['layer'], arrays['vector'], params['rows'], label, seen)
