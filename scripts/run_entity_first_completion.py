"""Finish saved generated neighbor branches without further intervention."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2) + '\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(destination, repo_path):
    root, repo = Path(destination), Path(repo_path)
    prior = root.parent / 'entity-first-neighbors-20261022'
    old, result = read(prior/'session.json'), read(prior/'summary.json')
    assert read(prior/'verification.json')['status'] == 'verified_complete'
    binary = old['binary']; assert sha(Path(binary)) == old['binary_sha256']
    vindex = '/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex'
    tok = Tokenizer.from_file(vindex+'/tokenizer.json')
    (root/'jobs').mkdir(parents=True, exist_ok=False)
    plans = []
    for variant in result['variants']:
        for branch, name, cap in [('edited', variant['committed_jobs'][-1], 8), ('baseline', variant['boundary_job'], 4)]:
            path = prior/'results'/f'{name}.json'
            source = read(path); job = read(prior/'jobs'/f'{name}.json')
            assert job['prompt_ids'] == old['prompts'][variant['index']]
            plans.append(dict(index=variant['index'], branch=branch, source_job=name,
                source_result_sha256=sha(path), prompt_ids=job['prompt_ids'], seed_ids=source['generated_ids'], max_new_tokens=cap))
    write(root/'session.json', dict(start=time.time(), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        binary=binary,binary_sha256=old['binary_sha256'],previous=str(prior),max_forwards=24,max_runtime_seconds=1800,plans=plans,
        scope='unpatched completion of verified generated branches; no supplied textual answer or fresh intervention'))
    child = subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'24'])
    count=0; deadline=time.monotonic()+1740; summary=dict(status='running',branches=[])
    try:
        for plan in plans:
            seed=list(plan['seed_ids']); record=dict(index=plan['index'],branch=plan['branch'],jobs=[],eos=False)
            for _ in range(plan['max_new_tokens']):
                assert count<24 and seed[-1] not in [1,106]
                name=f"{count:03}-v{plan['index']}-{plan['branch']}"
                job=dict(prompt_ids=plan['prompt_ids'],seed_ids=seed,max_new_tokens=1)
                pending=root/'jobs'/f'{name}.pending'; write(pending,job);pending.rename(pending.with_suffix('.json'))
                path=root/'results'/f'{name}.json'
                while not path.exists():
                    if child.poll() is not None: raise RuntimeError(f'runner exited {child.returncode}')
                    if time.monotonic()>deadline: raise TimeoutError('completion deadline')
                    time.sleep(1)
                res=read(path); step=res['steps'][0];assert step['ordinal']==count and not step['applied']
                with np.load(root/'results'/step['artifact']) as arrays:
                    assert all(np.isfinite(x).all() for x in arrays.values())
                    assert int(arrays['logits'].argmax())==step['top1']
                assert res['generated_ids']==seed+[step['top1']]
                seed=res['generated_ids'];count+=1;record['jobs'].append(name)
                record.update(text=tok.decode(seed,skip_special_tokens=False),eos=seed[-1] in [1,106])
                print(name+': '+record['text'],flush=True)
                if record['eos']:break
            summary['branches'].append(record);write(root/'progress.json',summary)
        summary['status']='unpatched completions finished'
    except Exception as exc:
        summary.update(status='incomplete',error=repr(exc));raise
    finally:
        (root/'STOP').touch()
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        path=root/'events.jsonl';events=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        summary.update(attempted_forwards=sum(x['type']=='attempt' for x in events),completed_forwards=sum(x['type']=='complete' for x in events),child_returncode=child.returncode,end=time.time())
        write(root/'summary.json',summary)
    if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')


if __name__=='__main__': main(*sys.argv[1:])
