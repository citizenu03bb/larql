"""Test one further frozen-vector application at the formatted symbol boundary."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(destination,repo_path):
    root,repo=Path(destination),Path(repo_path)
    prior=root.parent/'entity-first-completion-20261023'
    neighbor=root.parent/'entity-first-neighbors-20261022'
    complete=read(prior/'summary.json');old=read(neighbor/'session.json')
    assert read(prior/'verification.json')['status']=='verified_complete'
    assert all(x['eos'] for x in complete['branches'])
    vector=np.load(neighbor/'fixed-vector.npy');assert hashlib.sha256(vector.tobytes()).hexdigest()==old['vector_sha256']
    gain=old['gain'];assert gain==1
    binary=old['binary'];assert sha(Path(binary))==old['binary_sha256']
    vindex='/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex'
    tok=Tokenizer.from_file(vindex+'/tokenizer.json')
    plans=[]
    for branch in complete['branches']:
        if branch['branch']!='edited':continue
        source=branch['jobs'][0];result=read(prior/'results'/f'{source}.json');step=result['steps'][0]
        assert step['text'].strip()=='H'
        job=read(prior/'jobs'/f'{source}.json');index=branch['index'];position=old['positions'][index]
        assert job['prompt_ids'][position]==13076 and job['prompt_ids']==old['prompts'][index]
        plans.append(dict(index=index,source_job=source,source_result_sha256=sha(prior/'results'/f'{source}.json'),
            source_capture_sha256=sha(prior/'results'/step['artifact']),prompt_ids=job['prompt_ids'],seed_ids=job['seed_ids'],position=position))
    assert len(plans)==2
    (root/'jobs').mkdir(parents=True,exist_ok=False);np.save(root/'fixed-vector.npy',vector)
    write(root/'session.json',dict(start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        binary=binary,binary_sha256=old['binary_sha256'],previous=str(prior),neighbor=str(neighbor),max_forwards=24,max_runtime_seconds=1800,
        plans=plans,gain=gain,vector_sha256=old['vector_sha256'],scope='second application at formatted H; exploratory revised controller, not original single-trigger transfer'))
    child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'24'])
    count=0;deadline=time.monotonic()+1740;summary=dict(status='running',variants=[])
    def run(label,plan,seed,spec=None):
        nonlocal count
        assert count<24
        name=f'{count:03}-{label}';job=dict(prompt_ids=plan['prompt_ids'],seed_ids=list(seed),max_new_tokens=1)
        if spec is not None:job['intervention']=spec
        pending=root/'jobs'/f'{name}.pending';write(pending,job);pending.rename(pending.with_suffix('.json'))
        path=root/'results'/f'{name}.json'
        while not path.exists():
            if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('symbol retry deadline')
            time.sleep(1)
        result=read(path);step=result['steps'][0];assert step['ordinal']==count;count+=1
        z=dict(np.load(root/'results'/step['artifact']));assert all(np.isfinite(x).all() for x in z.values()) and int(z['logits'].argmax())==step['top1']
        print(name+': '+result['text'],flush=True);return step,z
    try:
        for plan in plans:
            index=plan['index'];seed=plan['seed_ids'];position=plan['position']
            source=read(prior/'results'/f"{plan['source_job']}.json")['steps'][0]
            historical=dict(np.load(prior/'results'/source['artifact']))
            baseline,b=run(f'v{index}-baseline',plan,seed)
            assert all(np.array_equal(b[k],x) for k,x in historical.items())
            spec=dict(layer=21,step=0,gain=0,row_deltas=[dict(position=position,delta=vector.tolist())])
            zero,z=run(f'v{index}-zero',plan,seed,spec)
            assert all(np.array_equal(b[k],x) for k,x in z.items())
            spec['gain']=gain;patched,p=run(f'v{index}-retry',plan,seed,spec)
            expected=b['prefix_layer21'].copy();expected[position]+=np.float32(gain)*vector
            assert np.array_equal(p['prefix_layer21'],expected)
            record=dict(index=index,baseline_job=baseline['job'],zero_job=zero['job'],intervention_job=patched['job'],symbol=patched['text'],committed_jobs=[patched['job']],eos=patched['top1'] in [1,106])
            generated=seed+[patched['top1']]
            for _ in range(9):
                if record['eos']:break
                step,_=run(f'v{index}-released',plan,generated)
                generated.append(step['top1']);record['committed_jobs'].append(step['job']);record['eos']=step['top1'] in [1,106]
            record['text']=tok.decode(generated,skip_special_tokens=False)
            summary['variants'].append(record);write(root/'progress.json',summary)
        summary['status']='formatted-symbol retries finished'
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
        summary.update(attempted_forwards=sum(x['type']=='attempt' for x in events),completed_forwards=sum(x['type']=='complete' for x in events),child_returncode=child.returncode,end=time.time());write(root/'summary.json',summary)
    if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')


if __name__=='__main__':main(*sys.argv[1:])
