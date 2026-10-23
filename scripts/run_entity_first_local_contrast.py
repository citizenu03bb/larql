"""Compare context-specific entity contrasts with the failed frozen-vector transfer."""
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
    prior=root.parent/'entity-first-neighbors-20261022'
    retry=root.parent/'entity-first-symbol-retry-20261023'
    assert read(retry/'verification.json')['status']=='verified_complete'
    retry_variants=read(retry/'summary.json')['variants']
    assert all(v['eos'] for v in retry_variants)
    failed_indices=[v['index'] for v in retry_variants if v['symbol'].strip()=='H']
    assert failed_indices==[0], 'scope frozen to the observed first-formulation failure'
    old=read(prior/'session.json');outcome=read(prior/'summary.json')
    assert read(prior/'verification.json')['status']=='verified_complete'
    binary=old['binary'];assert sha(Path(binary))==old['binary_sha256']
    vindex='/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex'
    tok=Tokenizer.from_file(vindex+'/tokenizer.json');frozen=np.load(prior/'fixed-vector.npy')
    plans=[]
    for v in outcome['variants']:
        if v['index'] not in failed_indices:continue
        index=v['index'];source=v['boundary_job'];job=read(prior/'jobs'/f'{source}.json');position=old['positions'][index]
        assert job['prompt_ids'][position]==13076
        donor=job['prompt_ids'].copy();donor[position]=42404
        plans.append(dict(index=index,source_job=source,source_result_sha256=sha(prior/'results'/f'{source}.json'),
            prompt_ids=job['prompt_ids'],donor_prompt_ids=donor,seed_ids=job['seed_ids'],position=position))
    (root/'jobs').mkdir(parents=True,exist_ok=False)
    write(root/'session.json',dict(start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        binary=binary,binary_sha256=old['binary_sha256'],previous=str(prior),retry=str(retry),plans=plans,gain=1,max_forwards=12,max_runtime_seconds=1800,
        scope='per-question hydrogen/helium contrast at original native H proposal; adaptive local editing, not fixed-vector transfer'))
    child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'12'])
    count=0;deadline=time.monotonic()+1740;summary=dict(status='running',variants=[])
    def run(label,prompt,seed,spec=None):
        nonlocal count
        assert count<12
        name=f'{count:03}-{label}';job=dict(prompt_ids=prompt,seed_ids=list(seed),max_new_tokens=1)
        if spec is not None:job['intervention']=spec
        pending=root/'jobs'/f'{name}.pending';write(pending,job);pending.rename(pending.with_suffix('.json'))
        path=root/'results'/f'{name}.json'
        while not path.exists():
            if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('local contrast deadline')
            time.sleep(1)
        result=read(path);step=result['steps'][0];assert step['ordinal']==count;count+=1
        arrays=dict(np.load(root/'results'/step['artifact']))
        assert all(np.isfinite(x).all() for x in arrays.values()) and int(arrays['logits'].argmax())==step['top1']
        print(name+': '+result['text'],flush=True);return step,arrays
    try:
        for plan in plans:
            index=plan['index'];prompt=plan['prompt_ids'];seed=plan['seed_ids'];position=plan['position']
            historical_step=read(prior/'results'/f"{plan['source_job']}.json")['steps'][0]
            historical=dict(np.load(prior/'results'/historical_step['artifact']))
            baseline,b=run(f'v{index}-baseline',prompt,seed)
            assert all(np.array_equal(b[k],x) for k,x in historical.items()) and baseline['text'].strip()=='H'
            donor,d=run(f'v{index}-donor',plan['donor_prompt_ids'],seed)
            delta=d['prefix_layer21']-b['prefix_layer21'];vector=delta[position].copy()
            contrast=f'v{index}-contrast.npz';np.savez_compressed(root/contrast,delta=delta,baseline=b['prefix_layer21'],donor=d['prefix_layer21'])
            spec=dict(layer=21,step=0,gain=0,row_deltas=[dict(position=position,delta=vector.tolist())])
            zero,z=run(f'v{index}-zero',prompt,seed,spec);assert all(np.array_equal(b[k],x) for k,x in z.items())
            spec['gain']=1;patched,p=run(f'v{index}-local',prompt,seed,spec)
            expected=b['prefix_layer21'].copy();expected[position]+=vector
            assert np.array_equal(p['prefix_layer21'],expected)
            a=vector.astype('float64');f=frozen.astype('float64')
            record=dict(index=index,baseline_job=baseline['job'],donor_job=donor['job'],zero_job=zero['job'],intervention_job=patched['job'],contrast=contrast,
                symbol=patched['text'],donor_symbol=donor['text'],vector_norm=float(np.linalg.norm(a)),frozen_cosine=float(a@f/(np.linalg.norm(a)*np.linalg.norm(f))),
                committed_jobs=[patched['job']],eos=patched['top1'] in [1,106])
            generated=seed+[patched['top1']]
            for _ in range(8):
                if record['eos']:break
                step,_=run(f'v{index}-released',prompt,generated)
                generated.append(step['top1']);record['committed_jobs'].append(step['job']);record['eos']=step['top1'] in [1,106]
            record['text']=tok.decode(generated,skip_special_tokens=False)
            summary['variants'].append(record);write(root/'progress.json',summary)
        summary['status']='context-specific contrasts finished'
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
