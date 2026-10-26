"""Verify saved NF4 bench evidence without loading a model (same author as runner).

Integrity checks and mechanistic outcomes are distinct. A valid negative is not
an integrity failure. Synthetic runs remain explicitly synthetic.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    summary = json.loads((root / 'summary.json').read_text())
    session = json.loads((root / 'session.json').read_text())
    assert summary['synthetic'] == session['synthetic']
    assert summary['status'] in ('complete', 'incomplete_native_cap'), summary.get('error')
    calls = summary['calls']
    assert len(calls) == summary['forwards'] <= 24
    assert [c['ordinal'] for c in calls] == list(range(len(calls)))
    saved, records = {}, {}
    for call in calls:
        p = root / call['artifact']
        assert p.parent == root and sha(p) == call['artifact_sha256']
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        assert all(np.isfinite(v).all() for v in arrays.values())
        assert call['argmax'] == int(arrays['logits'][-1].argmax())
        assert call['seconds'] > 0
        saved[call['name']], records[call['name']] = arrays, call
        if call['patch_layer'] is not None:
            key = f"layer{call['patch_layer']}"
            dtype = np.float32 if summary['synthetic'] else np.float16
            before, delta = arrays[key + '_before'].astype(dtype), arrays['requested_delta'].astype(dtype)
            assert np.array_equal((before + delta).astype(dtype).astype(np.float32), arrays[key + '_after'])
    prompt = session['prompt_ids']
    for name in ('warmup', 'timed0', 'timed1'):
        assert records[name]['input_ids'] == prompt and records[name]['patch_layer'] is None
        assert np.array_equal(saved['warmup']['logits'], saved[name]['logits'])
    native = summary['native']
    assert len(native['ids']) == len(native['call_names']) <= native['max_tokens']
    prefix = []
    for token, name in zip(native['ids'], native['call_names']):
        rec = records[name]
        assert rec['input_ids'] == prompt + prefix and rec['argmax'] == token
        assert rec['patch_layer'] is None
        assert not prefix or prefix[-1] not in session['eos_ids']
        prefix.append(token)
    assert native['eos'] == (prefix[-1] in session['eos_ids'])
    assert summary['status'] == ('complete' if native['eos'] else 'incomplete_native_cap')
    if not native['eos']:
        assert len(prefix) == native['max_tokens']
    timing = summary['timing']
    times = [records[x]['seconds'] for x in ('timed0', 'timed1')]
    assert times == timing['seconds'] and np.mean(times) == timing['mean_seconds']
    result = dict(status='verified_complete', synthetic=summary['synthetic'], forwards=len(calls),
        native_complete=native['eos'], native_text=native['text'],
        native_h_observed=bool(native['eos'] and re.search(r'\bH\b', native['text'])
                               and 'hydrogen' in native['text'].lower()),
        prefix_length=len(prompt), forward_mean_seconds=timing['mean_seconds'],
        verifier_author='same agent as runner; not independent verification')
    if summary['label'] == 'E2B':
        base = saved['boundary_base']
        kvkeys = [k for k in base if k.endswith(('_k', '_v'))]
        assert set(kvkeys) == {'sliding_attention_k', 'sliding_attention_v', 'full_attention_k', 'full_attention_v'}
        for name in ('boundary_repeat', 'boundary_zero12', 'boundary_zero14'):
            candidate = saved[name]
            assert all(np.array_equal(base[k], candidate[k]) for k in ['logits'] + kvkeys)
        delta = np.load(root / 'direction.npy', allow_pickle=False)
        spec = summary['boundary']
        v = np.random.default_rng(spec['seed']).standard_normal(base['layer14_before'].shape)
        target = 0.25 * np.linalg.norm(base['layer14_before'].astype(np.float64))
        expected = (v / np.linalg.norm(v) * target).astype(np.float32)
        assert np.array_equal(delta, expected) and target == spec['target_norm']
        conditions = {}
        boundary_ids = prompt + native['ids'][:1]
        for name, rec in records.items():
            if name.startswith('boundary_'):
                assert rec['input_ids'] == boundary_ids
                assert saved[name]['logits'].shape[0] == 2
                if rec['patch_layer'] is not None:
                    assert rec['position'] == len(prompt) - 1
        for layer in (12, 13, 14):
            name = f'boundary_patch{layer}'
            patched = saved[name]
            dtype = np.float32 if summary['synthetic'] else np.float16
            assert np.array_equal(patched['requested_delta'], delta.astype(dtype).astype(np.float32))
            assert np.array_equal(patched[f'layer{layer}_before'], base[f'layer{layer}_before'])
            difference = patched['logits'] - base['logits']
            conditions[str(layer)] = dict(
                current_max_abs=float(np.abs(difference[0]).max()),
                next_max_abs=float(np.abs(difference[1]).max()),
                next_bit_identical=np.array_equal(base['logits'][1], patched['logits'][1]),
                kv_max_abs={k: float(np.abs(patched[k] - base[k]).max()) for k in kvkeys},
                kv_bit_identical=all(np.array_equal(base[k], patched[k]) for k in kvkeys))
        late, early, mid = (conditions[str(i)] for i in (14, 12, 13))
        prediction = (late['current_max_abs'] > 0 and late['next_bit_identical'] and late['kv_bit_identical']
                      and early['next_max_abs'] > 0 and not early['kv_bit_identical']
                      and mid['next_max_abs'] > 0 and not mid['kv_bit_identical'])
        result.update(boundary_prediction_passed=bool(prediction), boundary_conditions=conditions,
                      boundary_replay_and_zeros_bit_identical=True,
                      direction_norm=float(np.linalg.norm(delta.astype(np.float64))))
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1])
    result = verify(root)
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
