"""Locate donor information outside the single entity row, at fixed L21/gain 1."""
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
    prior=root.parent/'entity-first-local-contrast-20261023'
    old=read(prior/'session.json');outcome=read(prior/'summary.json');plan=old['plans'][0];previous=outcome['variants'][0]
    assert read(prior/'verification.json')['status']=='verified_complete' and previous['eos']
    assert previous['text']=="Hydrogen's chemical symbol is $\\text{H}$.<turn|>"
    binary=old['binary'];assert sha(Path(binary))==old['binary_sha256']
    prompt=plan['prompt_ids'];donor_prompt=plan['donor_prompt_ids'];seed=plan['seed_ids'];pos=plan['position'];n=len(prompt)+len(seed)
    delta=np.load(prior/previous['contrast'])['delta'];assert delta.shape==(n,2560)
    groups=[('prompt',list(range(pos,len(prompt)))),('post_entity_prompt',list(range(pos+1,len(prompt)))),
            ('answer',list(range(len(prompt),n))),('all_after_entity',list(range(pos,n)))]
    (root/'jobs').mkdir(parents=True,exist_ok=False);np.save(root/'delta.npy',delta)
    write(root/'session.json',dict(start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        binary=binary,binary_sha256=old['binary_sha256'],previous=str(prior),prompt_ids=prompt,donor_prompt_ids=donor_prompt,seed_ids=seed,
        source_baseline_job=previous['baseline_job'],source_donor_job=previous['donor_job'],contrast_sha256=sha(prior/previous['contrast']),
        groups=groups,gain=1,max_forwards=15,max_runtime_seconds=1800,
        selection='first immediate He in declared group order; otherwise first group as formatting diagnostic',
        scope='multi-row L21 ablation; PLE inputs stay native even for all L21 rows; temporary local activation intervention'))
    vindex='/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex';tok=Tokenizer.from_file(vindex+'/tokenizer.json')
    child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'15'])
    count=0;deadline=time.monotonic()+1740;summary=dict(status='running',candidates=[])
    def run(label,ids,generated,spec=None):
        nonlocal count
        assert count<15
        name=f'{count:03}-{label}';job=dict(prompt_ids=ids,seed_ids=list(generated),max_new_tokens=1)
        if spec is not None:job['intervention']=spec
        pending=root/'jobs'/f'{name}.pending';write(pending,job);pending.rename(pending.with_suffix('.json'));path=root/'results'/f'{name}.json'
        while not path.exists():
            if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('distributed diagnostic deadline')
            time.sleep(1)
        result=read(path);step=result['steps'][0];assert step['ordinal']==count;count+=1
        z=dict(np.load(root/'results'/step['artifact']));assert all(np.isfinite(x).all() for x in z.values()) and int(z['logits'].argmax())==step['top1']
        print(name+': '+result['text'],flush=True);return step,z
    def source(name):
        step=read(prior/'results'/f'{name}.json')['steps'][0];return dict(np.load(prior/'results'/step['artifact']))
    def spec(positions,gain=1):return dict(layer=21,step=0,gain=gain,row_deltas=[dict(position=p,delta=delta[p].tolist()) for p in positions])
    try:
        baseline,b=run('baseline',prompt,seed);assert all(np.array_equal(b[k],x) for k,x in source(previous['baseline_job']).items())
        donor,d=run('donor',donor_prompt,seed);assert all(np.array_equal(d[k],x) for k,x in source(previous['donor_job']).items())
        zero,z=run('zero',prompt,seed,spec(groups[-1][1],0));assert all(np.array_equal(b[k],x) for k,x in z.items())
        summary.update(baseline_job=baseline['job'],donor_job=donor['job'],zero_job=zero['job'])
        selected=None
        for name,positions in groups:
            step,z=run(name,prompt,seed,spec(positions))
            expected=b['prefix_layer21'].copy();expected[positions]+=delta[positions]
            assert np.array_equal(expected,z['prefix_layer21'])
            candidate=dict(group=name,positions=positions,job=step['job'],token=step['text'],top1=step['top1'],
                donor_L21_max_abs_difference=float(np.max(np.abs(z['prefix_layer21']-d['prefix_layer21']))),
                donor_logits_max_abs_difference=float(np.max(np.abs(z['logits']-d['logits']))))
            summary['candidates'].append(candidate)
            if selected is None and step['text'].strip()=='He':selected=candidate
            write(root/'progress.json',summary)
        selected=selected or summary['candidates'][0];summary['selected']=selected
        generated=seed+[selected['top1']];chain=[selected['job']];eos=generated[-1] in [1,106]
        for _ in range(8):
            if eos:break
            step,_=run('released',prompt,generated);generated.append(step['top1']);chain.append(step['job']);eos=step['top1'] in [1,106]
        summary.update(status='distributed-state diagnostic finished',committed_jobs=chain,text=tok.decode(generated,skip_special_tokens=False),eos=eos)
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
