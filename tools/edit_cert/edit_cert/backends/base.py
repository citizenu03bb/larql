"""Edit backend protocol and on-disk format.

An edit lives in a directory: `backend.json` (kind, label, form, provenance, params) plus optional `.npy`
arrays. `content_sha256()` hashes what the edit *does* (kind, params, array bytes), not file bytes, so a
save/load round trip can be compared (control C7).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np

KINDS: dict[str, type] = {}


def register(cls):
    KINDS[cls.kind] = cls
    return cls


class Backend:
    kind = 'abstract'

    def __init__(self, label: str = '', seen_phrasings: list[str] | None = None):
        self.label = label
        self.seen_phrasings = list(seen_phrasings or [])

    # -- to implement
    def form(self) -> dict:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def arrays(self) -> dict[str, np.ndarray]:
        return {}

    def apply(self, inst) -> contextlib.AbstractContextManager:
        raise NotImplementedError

    def random_counterpart(self, seed: int) -> 'Backend':
        raise NotImplementedError

    @classmethod
    def from_parts(cls, label, seen, params, arrays) -> 'Backend':
        raise NotImplementedError

    # -- shared
    def content_sha256(self) -> str:
        h = hashlib.sha256(json.dumps(dict(kind=self.kind, params=self.params()), sort_keys=True).encode())
        for name, arr in sorted(self.arrays().items()):
            a = np.ascontiguousarray(arr)
            h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        return h.hexdigest()

    def save(self, path: Path) -> dict:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        for name, arr in self.arrays().items():
            np.save(path / f'{name}.npy', arr, allow_pickle=False)
        meta = dict(kind=self.kind, label=self.label, form=self.form(), seen_phrasings=self.seen_phrasings,
                    params=self.params(), arrays=sorted(self.arrays()), content_sha256=self.content_sha256())
        (path / 'backend.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
        return meta


def load(path: Path) -> Backend:
    path = Path(path)
    meta = json.loads((path / 'backend.json').read_text())
    cls = KINDS[meta['kind']]
    arrays = {n: np.load(path / f'{n}.npy', allow_pickle=False) for n in meta['arrays']}
    backend = cls.from_parts(meta['label'], meta['seen_phrasings'], meta['params'], arrays)
    if meta.get('content_sha256') and backend.content_sha256() != meta['content_sha256']:
        raise ValueError(f'{path}: content hash mismatch on load')
    return backend


def same_norm_gaussian(like: np.ndarray, seed: int) -> np.ndarray:
    """Gaussian draw scaled to the float16 norm of `like` (the NF4 bench computes in float16)."""
    target = float(np.linalg.norm(like.astype(np.float16).astype(np.float64)))
    draw = np.random.default_rng(seed).standard_normal(like.shape)
    return (draw / np.linalg.norm(draw) * target).astype(np.float32)
