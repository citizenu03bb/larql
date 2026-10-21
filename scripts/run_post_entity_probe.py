"""Post-entity search with conditional identity probes; at most 24 forwards."""
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
GROUPS = [('post-entity', list(range(15,25))), ('prompt-tail', list(range(18,31))), ('answer-preamble', list(range(31,37)))]
GAINS = [1, 2, 4]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(destination, repo_path):
    root, repo = Path(destination), Path(repo_path)
    previous = root.parent / 'neutral-boundary-transfer-20261021'
    old = read(previous / 'session.json')
    binary = old['binary']
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == old['binary_sha256']
    (root / 'jobs').mkdir(parents=True, exist_ok=False)
    (root / 'references').mkdir()
    historical = root.parent / 'preamble-transfer-20260928'
    target = read(historical / 'jobs/001-natural-reference.json')
    result = read(historical / 'results/001-natural-reference.json')
    shutil.copy2(historical / 'results' / result['steps'][0]['artifact'], root / 'references/natural-baseline.npz')
    tok = Tokenizer.from_file(str(VINDEX / 'tokenizer.json'))
    target_text = tok.decode(target['prompt_ids'], skip_special_tokens=False)
    donor_ids = tok.encode(target_text.replace('hydrogen', 'helium'), add_special_tokens=False).ids
    assert len(donor_ids) == len(target['prompt_ids']) == 31 and len(target['seed_ids']) == 6
    assert [(i, a, b) for i, (a, b) in enumerate(zip(target['prompt_ids'], donor_ids)) if a != b] == [(14, 13076, 42404)]
    assert target['prompt_ids'][17] == target['prompt_ids'][23] == 5404
    donor = dict(prompt_ids=donor_ids, seed_ids=target['seed_ids'], max_new_tokens=1)
    write(root / 'design.json', dict(target=target_text, donor=tok.decode(donor_ids, skip_special_tokens=False),
                                   differing_position=14, layer=21, groups=GROUPS, gains=GAINS,
                                   selection='first He candidate with hydrogen top1 in fixed name probe; stop search at such candidate',
                                   warning='donor changes the entity; He alone does not establish entity-preserving factual editing'))
    write(root / 'session.json', dict(start=time.time(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), binary=binary,
        binary_sha256=old['binary_sha256'], max_forwards=24, initial_forwards=3,
        max_candidates=9, conditional_identity_forwards=9, max_runtime_seconds=1800,
        scope='same natural question, native entity contrast, supplied six-token preamble; temporary activations only'))
    child = subprocess.Popen([sys.executable, '-u', str(repo / 'scripts/explore_natural_fact_edit.py'), str(root), binary, str(VINDEX), '24'])
    deadline = time.monotonic() + 1740
    summary = dict(status='running', trials=[], checks=[])
    count = 0

    def run(name, job):
        nonlocal count
        assert count + job['max_new_tokens'] <= 24
        write(root / 'jobs' / f'{name}.pending', job)
        (root / 'jobs' / f'{name}.pending').rename(root / 'jobs' / f'{name}.json')
        path = root / 'results' / f'{name}.json'
        while not path.exists():
            if child.poll() is not None:
                raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic() > deadline:
                raise TimeoutError('target-context diagnostic deadline')
            time.sleep(1)
        result = read(path)
        arrays = []
        for step in result['steps']:
            assert step['ordinal'] == count
            count += 1
            z = dict(np.load(root / 'results' / step['artifact']))
            assert all(np.isfinite(x).all() for x in z.values())
            assert int(z['logits'].argmax()) == step['top1']
            arrays.append(z)
        print(name + ': ' + result['text'], flush=True)
        return result, arrays

    try:
        rb, (base,) = run('000-target-baseline', target)
        expected = dict(np.load(root / 'references/natural-baseline.npz'))
        assert all(np.array_equal(base[k], v) for k, v in expected.items())
        assert rb['generated_ids'][-1] == 236814
        summary['checks'].append(dict(job='000-target-baseline', historical_arrays_bit_equal=True))
        rd, (on,) = run('001-native-helium-donor', donor)
        assert rd['generated_ids'][-1] == 2209
        delta = on['prefix_layer21'] - base['prefix_layer21']
        assert delta.shape == (37, 2560) and np.all(delta[:14] == 0)
        np.savez_compressed(root / 'native-L21-contrast.npz', delta=delta,
                            baseline=base['prefix_layer21'], donor=on['prefix_layer21'])
        def spec(positions, gain):
            return dict(layer=21, step=6, gain=gain, row_deltas=[dict(position=p, delta=delta[p].tolist()) for p in positions])
        zero = spec(list(range(14, 25)), 0)
        _, (identity,) = run('002-zero-gain-control', dict(target, intervention=zero))
        assert all(np.array_equal(identity[k], v) for k, v in base.items())
        summary['checks'].append(dict(job='002-zero-gain-control', all_arrays_bit_equal=True))
        identity_job = read(root.parent / 'relation-probe-20260927/jobs/010-entity-he-off.json')
        identity_job.pop('intervention', None)
        identity_job.pop('note', None)
        _, (namebase,) = run('003-identity-baseline', identity_job)
        summary['name_baseline'] = dict(token_id=int(namebase['logits'].argmax()), hydrogen_minus_helium=float(namebase['logits'][13076]-namebase['logits'][42404]))
        ordinal = 4
        selected = None
        for gain in GAINS:
            if selected is not None:
                break
            for label, positions in GROUPS:
                name = f'{ordinal:03}-{label}-gain{gain}'
                intervention = spec(positions, gain)
                result, (z,) = run(name, dict(target, intervention=intervention))
                ordinal += 1
                wanted = base['prefix_layer21'].copy()
                wanted[positions] += np.float32(gain) * delta[positions]
                assert np.array_equal(z['prefix_layer21'], wanted)
                assert np.array_equal(z['prefix_layer21'][14], base['prefix_layer21'][14])
                trial = dict(job=name, positions=positions, gain=gain, token=result['steps'][0]['text'],
                             He_minus_H=float(z['logits'][2209]-z['logits'][236814]), exact_L21_application=True)
                summary['trials'].append(trial)
                if result['generated_ids'][-1] == 2209:
                    name_result, (nz,) = run(f'{ordinal:03}-name-{label}-gain{gain}', dict(identity_job, intervention=intervention))
                    ordinal += 1
                    name_wanted = namebase['prefix_layer21'].copy()
                    name_wanted[positions] += np.float32(gain) * delta[positions]
                    assert np.array_equal(nz['prefix_layer21'],name_wanted)
                    trial['name_probe'] = dict(job=name_result['job'],token=name_result['steps'][0]['text'],hydrogen_minus_helium=float(nz['logits'][13076]-nz['logits'][42404]))
                    if name_result['generated_ids'][-1] == 13076:
                        selected = dict(trial=trial, intervention=intervention)
                        summary['selected'] = selected
                write(root / 'progress.json', summary)
                if selected is not None:
                    break
        summary['status'] = 'symbol and fixed name probe candidate selected' if selected else 'no candidate passed both symbol and fixed name probe'
        summary['scope_note'] = 'fixed name prefix is diagnostic only; free continuation remains to be tested'
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
        event_path = root / 'events.jsonl'
        events = [json.loads(x) for x in event_path.read_text().splitlines()] if event_path.exists() else []
        summary.update(attempted_forwards=sum(e['type']=='attempt' for e in events),
                       completed_forwards=sum(e['type']=='complete' for e in events),
                       child_returncode=child.returncode, end=time.time())
        write(root / 'summary.json', summary)
    if child.returncode:
        raise RuntimeError(f'runner failed {child.returncode}')


if __name__ == '__main__':
    main(*sys.argv[1:])
