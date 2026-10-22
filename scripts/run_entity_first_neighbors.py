"""Reuse a fixed successful entity delta on two paraphrases and two trigger controls."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

VINDEX=Path('/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex')
QUESTIONS=[
    "Give hydrogen's chemical symbol in one short sentence, with the element's name before the symbol.",
    "What symbol is used for hydrogen? Write one short sentence, placing the name of the element before its symbol.",
]


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def main(destination,repo_path):
    root,repo=Path(destination),Path(repo_path)
    prior=root.parent/'entity-first-probe-20261022'
    old=read(prior/'session.json');outcome=read(prior/'summary.json');verified=read(prior/'verification.json')
    assert verified['status']=='verified_complete' and outcome['edited_eos'] and outcome['baseline_eos']
    binary=old['binary'];assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==old['binary_sha256']
    vector=np.load(prior/'boundary-contrast.npz')['delta'][6].copy();gain=outcome['selected']['gain']
    tok=Tokenizer.from_file(str(VINDEX/'tokenizer.json'))
    def encode(q):return tok.encode('<bos><|turn>user\n'+q+'<turn|>\n<|turn>model\n',add_special_tokens=False).ids
    prompts=[encode(q) for q in QUESTIONS]
    positions=[]
    for ids in prompts:
        matches=[i for i,x in enumerate(ids) if x==13076];assert len(matches)==1;positions.append(matches[0])
    (root/'jobs').mkdir(parents=True,exist_ok=False)
    np.save(root/'fixed-vector.npy',vector)
    write(root/'session.json',dict(start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=binary,binary_sha256=old['binary_sha256'],previous=str(prior),max_forwards=24,max_runtime_seconds=1800,questions=QUESTIONS,prompts=prompts,positions=positions,gain=gain,vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),scope='two free-generation paraphrases with fixed vector/gain; helium/oxygen are supplied-prefix no-trigger identity controls'))
    child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,str(VINDEX),'24'])
    count=0;deadline=time.monotonic()+1740;summary=dict(status='running',variants=[],controls=[])
    def run(label,ids,generated,spec=None):
        nonlocal count
        assert count<24
        name=f'{count:03}-{label}';job=dict(prompt_ids=ids,seed_ids=list(generated),max_new_tokens=1)
        if spec is not None:job['intervention']=spec
        write(root/'jobs'/f'{name}.pending',job);(root/'jobs'/f'{name}.pending').rename(root/'jobs'/f'{name}.json')
        path=root/'results'/f'{name}.json'
        while not path.exists():
            if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('neighbor diagnostic deadline')
            time.sleep(1)
        result=read(path);step=result['steps'][0];assert step['ordinal']==count;count+=1
        z=dict(np.load(root/'results'/step['artifact']));assert all(np.isfinite(x).all() for x in z.values()) and int(z['logits'].argmax())==step['top1']
        print(name+': '+result['text'],flush=True);return step,z
    try:
        # Each variant gets ten forwards. This keeps four forwards for controls.
        for index,(ids,position) in enumerate(zip(prompts,positions)):
            limit=count+10;generated=[];record=dict(index=index,question=QUESTIONS[index],committed_jobs=[],injected=False,eos=False)
            while count<limit:
                native,z=run(f'v{index}-native',ids,generated)
                before=tok.decode(generated,skip_special_tokens=False)
                step=native
                if not record['injected'] and native['text'].strip()=='H' and 'hydrogen' in before.lower():
                    record['boundary_job']=native['job']
                    if count==limit:
                        record['budget_blocked_injection']=True
                    else:
                        spec=dict(layer=21,step=0,gain=gain,row_deltas=[dict(position=position,delta=vector.tolist())])
                        step,patched=run(f'v{index}-intervention',ids,generated,spec)
                        expected=z['prefix_layer21'].copy();expected[position]+=np.float32(gain)*vector
                        assert np.array_equal(patched['prefix_layer21'],expected)
                        record.update(injected=True,intervention_job=step['job'],symbol=step['text'],exact_application=True)
                generated.append(step['top1']);record['committed_jobs'].append(step['job'])
                record['text']=tok.decode(generated,skip_special_tokens=False)
                if step['top1'] in [1,106]:record['eos']=True;break
            summary['variants'].append(record);write(root/'progress.json',summary)
        for element in ['helium','oxygen']:
            q=old['question'].replace('hydrogen',element);ids=encode(q)
            prefix=outcome['boundary']['prefix'].replace('Hydrogen',element.capitalize()).replace('hydrogen',element)
            generated=tok.encode(prefix,add_special_tokens=False).ids
            eligible='hydrogen' in prefix.lower() and 13076 in ids
            assert not eligible
            baseline,b=run(f'{element}-baseline',ids,generated)
            controlled,c=run(f'{element}-controller-off',ids,generated)
            assert all(np.array_equal(b[k],c[k]) for k in b)
            summary['controls'].append(dict(element=element,prefix=prefix,eligible=eligible,baseline_job=baseline['job'],controller_job=controlled['job'],token=controlled['text'],all_arrays_bit_equal=True))
        summary['status']='fixed-vector paraphrases and trigger controls completed'
    except Exception as exc:
        summary.update(status='incomplete',error=repr(exc));raise
    finally:
        (root/'STOP').touch()
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        ep=root/'events.jsonl';events=[json.loads(x) for x in ep.read_text().splitlines()] if ep.exists() else []
        summary.update(attempted_forwards=sum(x['type']=='attempt' for x in events),completed_forwards=sum(x['type']=='complete' for x in events),child_returncode=child.returncode,end=time.time());write(root/'summary.json',summary)
    if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')


if __name__=='__main__':main(*sys.argv[1:])
