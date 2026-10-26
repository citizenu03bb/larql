"""Low-rank weight deltas added at run time to linear modules, without requantising them.

For each target: y = W x + scale * U (V^T x), with U (out x r) and V (in x r) kept in float32 and cast to the
activation dtype. Rank 1 is a ROME-style edit stored as a bf16-sized adapter; r > 1 covers LoRA-style deltas.
"""
from __future__ import annotations

import contextlib

import numpy as np

from .base import Backend, register


@register
class LowRank(Backend):
    kind = 'low_rank'

    def __init__(self, targets: list[dict], arrays: dict[str, np.ndarray], label='', seen_phrasings=None):
        """targets: [{module: 'model.language_model.layers.8.mlp.down_proj', layer: 8, scale: 1.0}], arrays
        U{i}/V{i} per target index."""
        super().__init__(label, seen_phrasings)
        self.targets = [dict(module=t['module'], layer=int(t['layer']), scale=float(t.get('scale', 1.0))) for t in targets]
        self._arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        for i, _ in enumerate(self.targets):
            u, v = self._arrays[f'U{i}'], self._arrays[f'V{i}']
            assert u.ndim == v.ndim == 2 and u.shape[1] == v.shape[1], (u.shape, v.shape)

    @property
    def rank(self):
        return max(self._arrays[f'U{i}'].shape[1] for i in range(len(self.targets)))

    def form(self):
        return dict(answer_time_trigger=False, donor_pass=False, uses_native_proposal=False,
                    substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=True,
                    max_write_layer=max(t['layer'] for t in self.targets), rank=self.rank)

    def params(self):
        return dict(targets=self.targets)

    def arrays(self):
        return dict(self._arrays)

    @contextlib.contextmanager
    def apply(self, inst):
        import torch
        handles = []
        for i, t in enumerate(self.targets):
            module = inst.model.get_submodule(t['module'])
            u = torch.as_tensor(self._arrays[f'U{i}'])
            v = torch.as_tensor(self._arrays[f'V{i}'])

            def post(mod, args, output, u=u, v=v, scale=t['scale']):
                x = args[0]
                uu, vv = u.to(x.device), v.to(x.device)
                delta = (x.float() @ vv) @ uu.T * scale
                return output + delta.to(output.dtype)
            handles.append(module.register_forward_hook(post))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def random_counterpart(self, seed):
        """Random U, V with the same column norms, so ||U V^T||_F matches for rank 1."""
        rng = np.random.default_rng(seed)
        out = {}
        for k, a in self._arrays.items():
            draw = rng.standard_normal(a.shape)
            draw *= np.linalg.norm(a, axis=0, keepdims=True) / np.linalg.norm(draw, axis=0, keepdims=True)
            out[k] = draw.astype(np.float32)
        return LowRank(self.targets, out, label=f'random({self.label}, seed={seed})')

    @classmethod
    def from_parts(cls, label, seen, params, arrays):
        return cls(params['targets'], arrays, label, seen)
