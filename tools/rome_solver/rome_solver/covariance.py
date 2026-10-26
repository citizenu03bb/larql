"""Fixed local English sample and uncentered second moment, as used by ROME.

No dataset hub API or download. Reading/tokenizing text is allowed in B0;
collect_model_covariances is reserved for approved B1 services.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .solver import Budget, frozen_model, site


SAMPLE_SEED = 20261026
QUOTAS = {"docs": 40, "emotion": 80, "banking77": 80}
WINDOW = 512
POSITIONS = tuple(range(7, WINDOW, 8))
RIDGE = 1e-4
DATA_FILES = {
    "emotion": "dair-ai___emotion/split/0.0.0/cab853a1dbdf4c42c2b3ef2173804746df8825fe/emotion-train.arrow",
    "banking77": "legacy-datasets___banking77/default/0.0.0/f54121560de48f2852f90be299010d1d6dc612ec/banking77-train.arrow",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def local_records(repo, datasets):
    """English ADRs plus exactly the two named cached training files; fail closed."""
    import pyarrow as pa

    repo, datasets = Path(repo), Path(datasets)
    docs = sorted((repo / "docs/adr").glob("*.md"))
    if not docs:
        raise FileNotFoundError("No English ADR documents")
    records = {"docs": [], "emotion": [], "banking77": []}
    sources = []
    for p in docs:
        if not p.resolve().is_relative_to(repo.resolve()):
            raise ValueError("ADR source escapes repository")
        data = p.read_bytes()
        name = p.relative_to(repo).as_posix()
        records["docs"].append((name, data.decode("utf-8")))
        sources.append(dict(source="docs", path=name, sha256=digest(data)))
    for source, relative in DATA_FILES.items():
        p = datasets / relative
        with pa.memory_map(str(p), "r") as f:
            table = pa.ipc.open_stream(f).read_all()
        texts = table.column("text").to_pylist()
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError(f"Empty/invalid text in {source}")
        records[source] = [(str(i), text) for i, text in enumerate(texts)]
        sources.append(dict(source=source, path=relative, sha256=digest(p.read_bytes()), rows=len(texts)))
    return records, sources


def fixed_windows(records, tokenizer, quotas=None, window=WINDOW, seed=SAMPLE_SEED):
    """Shuffle records, then sample disjoint full windows without replacement.

    All IDs, token bytes, source-stream hashes and offsets are fixed before
    any covariance forward. EOS separates documents; no padding or retries.
    """
    quotas = QUOTAS if quotas is None else quotas
    if window < 1 or not quotas or tokenizer.eos_token_id is None:
        raise ValueError("Invalid window plan")
    rng = np.random.default_rng(seed)
    windows, entries, streams = [], [], {}
    for source, count in quotas.items():
        rows = sorted(records[source], key=lambda row: row[0])
        if len({r[0] for r in rows}) != len(rows) or count < 1:
            raise ValueError("Duplicate IDs/invalid quota")
        order = rng.permutation(len(rows))
        ids = []
        for i in order:
            ids.extend(tokenizer.encode(rows[i][1], add_special_tokens=False))
            ids.append(tokenizer.eos_token_id)
        stream = np.asarray(ids, dtype="<i8")
        available = len(stream) // window
        if available < count:
            raise ValueError(f"Not enough {source} full windows: {available} < {count}")
        starts = sorted(int(i) * window for i in rng.choice(available, size=count, replace=False))
        streams[source] = dict(records=len(rows), tokens=len(stream), sha256=digest(stream.tobytes()))
        for start in starts:
            block = stream[start:start + window].copy()
            entries.append(dict(index=len(windows), source=source, start=start, sha256=digest(block.tobytes())))
            windows.append(block)
    array = np.stack(windows)
    return array, dict(seed=seed, window=window, quotas=dict(quotas), eos_id=tokenizer.eos_token_id,
                       streams=streams, windows=entries, tokens_sha256=digest(array.tobytes()),
                       sampling="sorted IDs; seeded record permutation; EOS separators; disjoint full windows")


class SecondMoment:
    def __init__(self, width):
        self.sum = np.zeros((width, width), dtype=np.float64)
        self.count = 0

    def add(self, keys):
        keys = np.asarray(keys, dtype=np.float64)
        if keys.ndim != 2 or keys.shape[1] != len(self.sum) or not np.isfinite(keys).all():
            raise ValueError("Invalid covariance keys")
        self.sum += keys.T @ keys
        self.count += len(keys)

    def finish(self, ridge=RIDGE):
        if not self.count or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Missing keys/invalid ridge")
        moment = self.sum / self.count
        scale = np.trace(moment) / len(moment)
        if not np.isfinite(moment).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Nonfinite/zero covariance")
        regularized = moment + np.eye(len(moment)) * ridge * scale
        return regularized.astype(np.float32), dict(count=self.count, ridge=float(ridge),
                                                   diagonal_added=float(ridge * scale),
                                                   statistic="uncentered E[k k^T]")


def collect_model_covariances(model, layers, windows, candidate_layers=(4, 8, 12),
                              label="E2B", synthetic=False, positions=POSITIONS,
                              ridge=RIDGE, budget=None):
    """B1 only: capture every candidate layer in the SAME full-prefix forward."""
    budget = budget or Budget()
    if len(candidate_layers) != len(set(candidate_layers)):
        raise ValueError("Duplicate candidate layer")
    windows = np.asarray(windows)
    if windows.ndim != 2 or windows.dtype.kind not in "iu" or not len(windows):
        raise ValueError("Invalid windows")
    positions = tuple(positions)
    if not positions or len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= windows.shape[1]:
        raise ValueError("Invalid covariance positions")
    modules = {i: site(model, layers, i, label, synthetic)[1] for i in candidate_layers}
    stats = {i: SecondMoment(m.in_features) for i, m in modules.items()}
    handles, calls = [], {}
    try:
        for i, module in modules.items():
            def pre(mod, args, i=i):
                calls[i] = calls.get(i, 0) + 1
                if calls[i] != 1:
                    raise ValueError("Covariance hook ran twice")
                stats[i].add(args[0][0, list(positions)].detach().float().cpu().numpy())
            handles.append(module.register_forward_pre_hook(pre))
        with frozen_model(model), torch.no_grad():
            device = next(layers.parameters()).device
            for window in windows:
                calls.clear()
                budget.take()
                out = model(input_ids=torch.as_tensor(window[None], dtype=torch.long, device=device),
                            use_cache=False, logits_to_keep=1)
                if set(calls) != set(modules) or not torch.isfinite(out.logits).all():
                    raise ValueError("Incomplete covariance hooks/nonfinite output")
    finally:
        for handle in handles:
            handle.remove()
    results, metadata = {}, {}
    for layer, stat in stats.items():
        results[layer], metadata[layer] = stat.finish(ridge)
    return results, dict(layers=metadata, positions=list(positions), forwards=budget.forwards,
                         gradient_steps=budget.gradient_steps)
