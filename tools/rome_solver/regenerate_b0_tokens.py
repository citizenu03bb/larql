#!/usr/bin/env python3
"""Regenerate withdrawn B0 tokens privately from checksum-pinned local inputs.

Tokenizer only: no pretrained weights, model forward, dataset download or
license permission is supplied by this script. Output must be outside this worktree.
The historical sample/manifest/runner stay unchanged.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / 'docs/research-data/phase-b0-preflight/covariance-sample.json'
FROZEN = REPO / 'docs/research-data/phase-b0-manifest.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def regenerate(datasets, tokenizer_root, out, docs_revision=None):
    out = out.resolve()
    require(not out.is_relative_to(REPO.resolve()), 'Keep the withdrawn token file outside this Git worktree')
    require(not out.exists(), 'Output already exists; no overwrite')
    frozen = json.loads(FROZEN.read_text()); sample = json.loads(SAMPLE.read_text())
    require(sha(SAMPLE.read_bytes()) == frozen['files'][str(SAMPLE.relative_to(REPO))]['sha256'], 'Frozen sample receipt changed')
    helper = REPO / 'tools/rome_solver/rome_solver/covariance.py'
    require(sha(helper.read_bytes()) == frozen['files'][str(helper.relative_to(REPO))]['sha256'], 'Frozen sampling helper changed')
    for name, expected in sample['tokenizer_metadata_sha256'].items():
        require(sha((tokenizer_root / name).read_bytes()) == expected, 'Tokenizer metadata drift: ' + name)
    records = {'docs': [], 'emotion': [], 'banking77': []}; verified = []
    import pyarrow as pa
    for source in sample['sources']:
        kind, name = source['source'], source['path']
        if kind == 'docs':
            raw = subprocess.check_output(['git', 'show', docs_revision + ':' + name], cwd=REPO) if docs_revision else (REPO / name).read_bytes()
            require(sha(raw) == source['sha256'], 'ADR drift: ' + name + '; use --docs-revision with the historical source')
            records[kind].append((name, raw.decode('utf-8')))
        else:
            path = datasets / name
            require(sha(path.read_bytes()) == source['sha256'], 'Cached Arrow source drift: ' + name)
            with pa.memory_map(str(path), 'r') as stream:
                table = pa.ipc.open_stream(stream).read_all()
            texts = table.column('text').to_pylist()
            require(len(texts) == source['rows'], 'Training-row count changed')
            records[kind] = [(str(i), text) for i, text in enumerate(texts)]
        verified.append(dict(source=kind, path=name, sha256=source['sha256']))
    from transformers import AutoTokenizer
    from rome_solver.covariance import fixed_windows
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    # JSON sort_keys changed the receipt's quota key order. The original RNG
    # loop order is docs, emotion, banking77, and must be restored explicitly.
    quotas = {name: sample['quotas'][name] for name in ('docs', 'emotion', 'banking77')}
    tokens, rebuilt = fixed_windows(records, tokenizer, quotas=quotas, window=sample['window'], seed=sample['seed'])
    for name in ('seed', 'window', 'quotas', 'eos_id', 'streams', 'windows', 'tokens_sha256', 'sampling'):
        require(rebuilt[name] == sample[name], 'Regeneration differs: ' + name)
    require(tokens.shape == (200, 512) and tokens.dtype == np.dtype('<i8'), 'Unexpected reconstructed array')
    buffer = io.BytesIO(); np.save(buffer, tokens, allow_pickle=False); payload = buffer.getvalue()
    expected = sample['token_file_sha256']
    require(sha(payload) == expected == frozen['files']['docs/research-data/phase-b0-preflight/covariance-tokens.npy']['sha256'], 'Reconstructed .npy bytes do not match historical evidence')
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(payload)
    require(sha(out.read_bytes()) == expected, 'Read-back checksum mismatch')
    return dict(author='Codex', model_forwards=0, pretrained_weights_loaded=False,
                download_performed=False, output_outside_worktree=True,
                source_files_verified=len(verified), source_sha256=verified,
                shape=list(tokens.shape), dtype=str(tokens.dtype), bytes=len(payload),
                tokenizer_metadata_sha256=sample['tokenizer_metadata_sha256'],
                token_file_sha256=expected, tokens_sha256=sample['tokens_sha256'],
                all_200_window_hashes_match=True, all_stream_hashes_match=True,
                frozen_sample_unchanged=True, docs_revision=docs_revision,
                helper_sha256=sha(helper.read_bytes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, default=Path('/mnt/models/distill-course/hf/datasets'))
    p.add_argument('--tokenizer-root', type=Path, default=Path('/mnt/models/hf-models/gemma-4-E2B-it'))
    p.add_argument('--docs-revision', help='Optional historical git ref for only the hash-listed ADR sources')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--receipt', type=Path)
    args = p.parse_args()
    result = regenerate(args.datasets, args.tokenizer_root, args.out, args.docs_revision)
    if args.receipt:
        with args.receipt.open('x') as f:
            f.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: result[key] for key in ('model_forwards','bytes','token_file_sha256','all_200_window_hashes_match')}))


if __name__ == '__main__':
    main()
