"""Two-phase activation steering: symbol intervention then identity preservation."""
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
GROUPS = [('relation', [17]), ('entity', [14]), ('entity-relation', list(range(14, 25)))]
GAINS = [1, 2, 4, 8]


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
                                   differing_position=14, layer=21, symbol_positions=[14], symbol_gain=1, identity_gains=[-0.125,-0.25,-0.5,-1.0],
                                   selection='smallest tested negative entity gain predicting hydrogen at fixed name prefix',
                                   warning='temporal controller is a distinct intervention, not a static factual patch; all final generation tokens must be greedy outputs'))
    write(root / 'session.json', dict(start=time.time(), source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), binary=binary,
        binary_sha256=old['binary_sha256'], max_forwards=24, calibration_max_forwards=7,
        from_prompt_generation_max_forwards=15, max_runtime_seconds=1800,
        scope='calibration at supplied prefixes then generation from empty answer; temporary two-phase activation steering, no weight edit'))
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
        prior = root.parent / 'target-context-probe-20261021'
        contrast = np.load(prior / 'native-L21-contrast.npz')
        delta = contrast['delta'].copy()
        assert np.array_equal(contrast['baseline'],base['prefix_layer21'])
        np.savez_compressed(root / 'native-L21-contrast.npz',delta=delta,baseline=contrast['baseline'],donor=contrast['donor'])
        def spec(gain):
            return dict(layer=21,step=0,gain=gain,row_deltas=[dict(position=14,delta=delta[14].tolist())])
        positive = dict(target,intervention=spec(1))
        rp,(pz,) = run('001-symbol-positive',positive)
        old_positive = read(prior / 'results/004-entity-gain1.json')
        saved = np.load(prior / 'results' / old_positive['steps'][0]['artifact'])
        assert all(np.array_equal(pz[k],saved[k]) for k in saved.files) and rp['generated_ids'][-1]==2209
        identity_job = read(root.parent / 'relation-probe-20260927/jobs/010-entity-he-off.json')
        identity_job.pop('intervention',None);identity_job.pop('note',None)
        _,(namebase,) = run('002-name-baseline',identity_job)
        selected_gain=None
        summary['calibration']=[]
        for gain in [-0.125,-0.25,-0.5,-1.0]:
            r,(z,)=run(f'{count:03}-identity-gain{gain}',dict(identity_job,intervention=spec(gain)))
            expected=namebase['prefix_layer21'].copy();expected[14]+=np.float32(gain)*delta[14]
            assert np.array_equal(expected,z['prefix_layer21'])
            trial=dict(gain=gain,token=r['steps'][0]['text'],hydrogen_minus_helium=float(z['logits'][13076]-z['logits'][42404]))
            summary['calibration'].append(trial);write(root/'progress.json',summary)
            if r['generated_ids'][-1]==13076:
                selected_gain=gain;break
        summary['selected_identity_gain']=selected_gain
        if selected_gain is None:
            summary['status']='no tested identity gain preserved hydrogen at fixed prefix'
        else:
            # All initial output tokens are generated, never supplied by the experiment.
            preamble,_=run(f'{count:03}-free-preamble',dict(prompt_ids=target['prompt_ids'],seed_ids=[],max_new_tokens=6))
            summary['free_preamble']=preamble['text']
            if preamble['generated_ids']!=target['seed_ids']:
                summary['status']='native preamble differs from calibrated symbol boundary; stopped without injection'
            else:
                symbol,_=run(f'{count:03}-free-symbol',dict(prompt_ids=target['prompt_ids'],seed_ids=preamble['generated_ids'],max_new_tokens=1,intervention=spec(1)))
                assert symbol['generated_ids'][-1]==2209
                continuation,_=run(f'{count:03}-identity-continuation',dict(prompt_ids=target['prompt_ids'],seed_ids=symbol['generated_ids'],max_new_tokens=8,intervention=spec(selected_gain)))
                summary['continuation']=continuation['text']
                summary['hydrogen_in_continuation']='hydrogen' in continuation['text'].lower()
                summary['helium_in_continuation']='helium' in continuation['text'].lower()
                summary['status']='two-phase from-prompt generation tested'
                summary['phases']=dict(preamble='native greedy',symbol_gain=1,continuation_gain=selected_gain,forced_output_tokens=0)
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
