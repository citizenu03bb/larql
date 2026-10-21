"""Transfer a source-induced L22/KV delta to two position-aligned neutral prompts.

Eight fixed symbol/control forwards, then at most eight paired continuation
steps if a target chooses He. Reuse the immutable worker; never alter weights.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

VINDEX = Path('/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex')
PREFIXES = [
    'Please answer the following question clearly and concisely, using standard terminology and a complete sentence.',
    'Please answer this question accurately and clearly. Use standard terminology and write a complete sentence.',
]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(destination, repo_path):
    root, repo = Path(destination), Path(repo_path)
    prior = root.parent / 'kv-boundary-20260928'
    old = read(prior / 'session.json')
    binary = old['binary']
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == old['binary_sha256']
    (root / 'jobs').mkdir(parents=True, exist_ok=False)
    (root / 'references').mkdir()
    refs = {}
    for name, label in [('000-baseline', 'baseline'), ('001-positive', 'positive')]:
        job = read(prior / 'jobs' / f'{name}.json')
        result = read(prior / 'results' / f'{name}.json')
        shutil.copy2(prior / 'results' / result['steps'][0]['artifact'], root / 'references' / f'{label}.npz')
        write(root / 'references' / f'{label}-job.json', job)
        refs[label] = (job, dict(np.load(root / 'references' / f'{label}.npz')))
    source, base = refs['baseline']
    positive_job, positive = refs['positive']
    assert source['prompt_ids'] == positive_job['prompt_ids']
    assert source['seed_ids'] == positive_job['seed_ids'] and len(source['seed_ids']) == 6
    assert len(source['prompt_ids']) == 48
    delta = {k: positive[k] - base[k] for k in ['prefix_layer22', 'kv22_keys', 'kv22_values']}
    assert np.flatnonzero(np.any(delta['prefix_layer22'] != 0, axis=1)).tolist() == list(range(15, 54))
    # Prior localization establishes sufficiency without replacing answer rows.
    delta['prefix_layer22'][48:] = 0
    for key in ['kv22_keys', 'kv22_values']:
        assert np.flatnonzero(np.any(delta[key] != 0, axis=1)).tolist() == [15, 16, 17, 18]
    np.savez_compressed(root / 'transferred-deltas.npz', **delta)
    tok = Tokenizer.from_file(str(VINDEX / 'tokenizer.json'))
    targets = []
    for prefix in PREFIXES:
        prefix_ids = tok.encode(prefix, add_special_tokens=False).ids
        assert len(prefix_ids) == 17
        ids = source['prompt_ids'][:4] + prefix_ids + source['prompt_ids'][21:]
        text = tok.decode(ids, skip_special_tokens=False)
        assert tok.encode(text, add_special_tokens=False).ids == ids
        assert ids[21:] == source['prompt_ids'][21:] and len(ids) == 48
        targets.append(dict(prompt_ids=ids, seed_ids=source['seed_ids'], max_new_tokens=1))
    write(root / 'alignment.json', dict(
        method='same absolute positions; replace 17-token fictional prefix by 17-token neutral instruction; shared suffix unchanged',
        target_texts=[tok.decode(j['prompt_ids'], skip_special_tokens=False) for j in targets],
        target_prompt_ids=[j['prompt_ids'] for j in targets],
        residual_positions=list(range(15, 48)), kv_positions=[15, 16, 17, 18],
        gain=1, key_rotation='none: no position shift',
        limitation='prefix positions are aligned by coordinate, not by identical words; supplied six-token answer preamble'))
    write(root / 'session.json', dict(
        mode='exploratory neutral-prefix joint-delta transfer', max_forwards=24,
        max_runtime_seconds=1800, start=time.time(), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        binary=binary, binary_sha256=old['binary_sha256'], previous=str(prior),
        numpy=np.__version__, python=sys.version, prefixes=PREFIXES,
        scope='two neutral instructions at source length; symbol after supplied preamble; conditional eight-token continuation'))
    child = subprocess.Popen([sys.executable, '-u', str(repo / 'scripts/explore_natural_fact_edit.py'), str(root), binary, str(VINDEX), '24'])
    deadline = time.monotonic() + 1740
    summary = dict(status='running', conditions=[], checks=[], continuation=[])
    ordinal = 0

    def run(name, job):
        nonlocal ordinal
        assert ordinal < 24 and job['max_new_tokens'] == 1
        write(root / 'jobs' / f'{name}.pending', job)
        (root / 'jobs' / f'{name}.pending').rename(root / 'jobs' / f'{name}.json')
        path = root / 'results' / f'{name}.json'
        while not path.exists():
            if child.poll() is not None:
                raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic() > deadline:
                raise TimeoutError('neutral transfer deadline')
            time.sleep(1)
        result = read(path)
        step = result['steps'][0]
        arrays = dict(np.load(root / 'results' / step['artifact']))
        assert step['ordinal'] == ordinal
        ordinal += 1
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert int(arrays['logits'].argmax()) == step['top1']
        summary['conditions'].append(dict(job=name, token=step['text'], He_minus_H=float(arrays['logits'][2209] - arrays['logits'][236814])))
        print(name + ': ' + result['text'], flush=True)
        write(root / 'progress.json', summary)
        return result, arrays

    def intervene(name, job, native, mode):
        matrices = {}
        for key, change in delta.items():
            x = native[key].copy()
            if mode == 'joint' or (mode == 'residual' and key == 'prefix_layer22') or (mode == 'kv' and key != 'prefix_layer22'):
                # Only prompt rows, valid also for longer continuation inputs.
                x[:48] += change[:48]
            matrices[key] = x
        boundary = dict(residual=matrices['prefix_layer22'].tolist(), kv=dict(
            keys=matrices['kv22_keys'].tolist(), values=matrices['kv22_values'].tolist()))
        patched = dict(job, intervention=dict(layer=22, step=len(job['seed_ids']), gain=1, boundary=boundary))
        result, arrays = run(name, patched)
        assert np.array_equal(arrays['prefix_layer21'], native['prefix_layer21'])
        assert all(np.array_equal(arrays[k], v) for k, v in matrices.items())
        if mode == 'identity':
            assert all(np.array_equal(arrays[k], v) for k, v in native.items())
        summary['checks'].append(dict(job=name, exact_boundary=True, L21_unchanged=True, identity_bit_equal=mode == 'identity'))
        return result, arrays

    try:
        _, z = run('000-source-positive', positive_job)
        assert all(np.array_equal(z[k], v) for k, v in positive.items())
        summary['checks'].append(dict(job='000-source-positive', all_arrays_bit_equal=True))
        _, a = run('001-natural-A-baseline', targets[0])
        intervene('002-natural-A-identity', targets[0], a, 'identity')
        ra, _ = intervene('003-natural-A-joint', targets[0], a, 'joint')
        intervene('004-natural-A-residual', targets[0], a, 'residual')
        intervene('005-natural-A-kv', targets[0], a, 'kv')
        _, b = run('006-natural-B-baseline', targets[1])
        rb, _ = intervene('007-natural-B-joint', targets[1], b, 'joint')
        candidates = [(targets[0], ra, 'A'), (targets[1], rb, 'B')]
        success = next(((j, r, label) for j, r, label in candidates if r['generated_ids'][-1] == 2209), None)
        if success is None:
            summary['status'] = 'no He on either neutral prompt at gain 1; stopped after eight forwards'
        else:
            target, result, label = success
            generated = list(result['generated_ids'])
            summary['continuation_target'] = label
            for i in range(8):
                j = dict(prompt_ids=target['prompt_ids'], seed_ids=generated, max_new_tokens=1)
                # Capture native state at the exact current input length; never reuse
                # a shorter capture or rely on cross-length prefix equality.
                _, native = run(f'{8+2*i:03}-continuation-native-{i}', j)
                result, _ = intervene(f'{9+2*i:03}-continuation-joint-{i}', j, native, 'joint')
                generated = list(result['generated_ids'])
                summary['continuation'].append(dict(step=i, token=result['steps'][0]['text'], text=result['text']))
                if generated[-1] in [1, 106]:
                    break
            summary['status'] = 'natural symbol transfer and bounded continuation tested'
            summary['continuation_text'] = tok.decode(generated, skip_special_tokens=False)
    except Exception as exc:
        summary.update(status='incomplete', error=repr(exc))
        raise
    finally:
        (root / 'STOP').touch()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        events_path = root / 'events.jsonl'
        events = [json.loads(x) for x in events_path.read_text().splitlines()] if events_path.exists() else []
        summary.update(attempted_forwards=sum(e['type'] == 'attempt' for e in events),
                       completed_forwards=sum(e['type'] == 'complete' for e in events),
                       child_returncode=child.returncode, end=time.time())
        write(root / 'summary.json', summary)
    if child.returncode:
        raise RuntimeError(f'runner failed {child.returncode}')


if __name__ == '__main__':
    main(*sys.argv[1:])
