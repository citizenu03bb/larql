"""Two-by-two L21 support/norm comparison, with two new free continuations."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(destination,repo_path):
    root,repo=Path(destination),Path(repo_path)
    local=root.parent/'entity-first-local-contrast-20261023'
    wide=root.parent/'entity-first-distributed-20261023'
    for source in [local,wide]:assert read(source/'verification.json')['status']=='verified_complete'
    old=read(local/'session.json');lv=read(local/'summary.json')['variants'][0];wv=read(wide/'summary.json');plan=old['plans'][0]
    binary=old['binary'];assert sha(Path(binary))==old['binary_sha256']
    prompt,seed=plan['prompt_ids'],plan['seed_ids'];position=plan['position'];positions=list(range(position,len(prompt)))
    contrast=local/lv['contrast'];z=dict(np.load(contrast));delta=z['delta']
    assert np.array_equal(delta,z['donor']-z['baseline'])
    entity_norm=float(np.linalg.norm(delta[position].astype('float64')))
    prompt_norm=float(np.linalg.norm(delta[positions].astype('float64')))
    assert 0<entity_norm<prompt_norm
    conditions=[dict(name='entity_low',positions=[position],gain=1.,level='low',replay_root=str(local),replay_job=lv['intervention_job']),
                dict(name='prompt_high',positions=positions,gain=1.,level='high',replay_root=str(wide),replay_job=wv['selected']['job']),
                dict(name='entity_high',positions=[position],gain=float(np.float32(prompt_norm/entity_norm)),level='high'),
                dict(name='prompt_low',positions=positions,gain=float(np.float32(entity_norm/prompt_norm)),level='low')]
    for condition in conditions:
        effect=np.float32(condition['gain'])*delta[condition['positions']]
        actual=float(np.linalg.norm(effect.astype('float64')))
        target=entity_norm if condition['level']=='low' else prompt_norm
        assert abs(actual/target-1)<1e-6
        condition.update(injected_norm=actual,target_norm=target)
    (root/'jobs').mkdir(parents=True,exist_ok=False);np.save(root/'delta.npy',delta)
    write(root/'session.json',dict(start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        binary=binary,binary_sha256=old['binary_sha256'],local_source=str(local),wide_source=str(wide),contrast_sha256=sha(contrast),
        prompt_ids=prompt,seed_ids=seed,position=position,entity_norm=entity_norm,prompt_norm=prompt_norm,conditions=conditions,
        source_baseline_job=lv['baseline_job'],max_forwards=24,max_runtime_seconds=1800,
        scope='matched Frobenius norms at one observed generated boundary; two old conditions replayed, both new conditions continued without further patch'))
    vindex='/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex';tok=Tokenizer.from_file(vindex+'/tokenizer.json')
    child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'24'])
    count=0;deadline=time.monotonic()+1740;summary=dict(status='running',conditions=[],branches=[])
    def run(label,generated,spec=None):
        nonlocal count
        assert count<24
        name=f'{count:03}-{label}';job=dict(prompt_ids=prompt,seed_ids=list(generated),max_new_tokens=1)
        if spec is not None:job['intervention']=spec
        pending=root/'jobs'/f'{name}.pending';write(pending,job);pending.rename(pending.with_suffix('.json'));path=root/'results'/f'{name}.json'
        while not path.exists():
            if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('norm-matched diagnostic deadline')
            time.sleep(1)
        result=read(path);step=result['steps'][0];assert step['ordinal']==count;count+=1
        arrays=dict(np.load(root/'results'/step['artifact']))
        assert all(np.isfinite(x).all() for x in arrays.values()) and int(arrays['logits'].argmax())==step['top1']
        print(name+': '+result['text'],flush=True);return step,arrays
    def capture(source,name):
        step=read(source/'results'/f'{name}.json')['steps'][0];return dict(np.load(source/'results'/step['artifact']))
    def spec(rows,gain):return dict(layer=21,step=0,gain=gain,row_deltas=[dict(position=p,delta=delta[p].tolist()) for p in rows])
    try:
        baseline,b=run('baseline',seed);assert all(np.array_equal(b[k],x) for k,x in capture(local,lv['baseline_job']).items())
        zero,q=run('zero',seed,spec(positions,0));assert all(np.array_equal(b[k],x) for k,x in q.items())
        summary.update(baseline_job=baseline['job'],zero_job=zero['job'])
        for condition in conditions:
            rows,gain=condition['positions'],condition['gain']
            step,arrays=run(condition['name'],seed,spec(rows,gain))
            expected=b['prefix_layer21'].copy();expected[rows]+=np.float32(gain)*delta[rows]
            assert np.array_equal(expected,arrays['prefix_layer21'])
            if 'replay_root' in condition:
                oldarrays=capture(Path(condition['replay_root']),condition['replay_job'])
                assert all(np.array_equal(arrays[k],x) for k,x in oldarrays.items())
            record=dict(name=condition['name'],job=step['job'],top1=step['top1'],token=step['text'],
                he_minus_h=float(arrays['logits'][1293]-arrays['logits'][640]),exact_application=True,
                historical_replay_bit_equal='replay_root' in condition)
            summary['conditions'].append(record);write(root/'progress.json',summary)
        for condition in summary['conditions'][2:]:
            generated=seed+[condition['top1']];record=dict(name=condition['name'],committed_jobs=[condition['job']],eos=generated[-1] in [1,106])
            for _ in range(9):
                if record['eos']:break
                step,_=run(condition['name']+'-released',generated)
                generated.append(step['top1']);record['committed_jobs'].append(step['job']);record['eos']=step['top1'] in [1,106]
            record['text']=tok.decode(generated,skip_special_tokens=False)
            summary['branches'].append(record);write(root/'progress.json',summary)
        summary['status']='norm-matched factorial diagnostic finished'
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
