"""Integrated from-empty replay of the selected distributed-state intervention.

No supplied answer tokens and no logit/token forcing. At most 24 forwards.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

VINDEX = Path('/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex')
QUESTION = "Give hydrogen's chemical symbol in one short sentence, with the element's name before the symbol."


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main(destination, repo_path):
    root, repo = Path(destination), Path(repo_path)
    prior = root.parent / 'entity-first-distributed-20261023'
    old = read(prior / 'session.json')
    evidence = read(prior / 'verification.json')
    assert evidence['status'] == 'verified_complete' and evidence['eos']
    assert evidence['selected']['token'].strip() == 'He'
    group_name = evidence['selected']['group']
    binary = old['binary']
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == old['binary_sha256']
    (root / 'jobs').mkdir(parents=True, exist_ok=False)
    tok = Tokenizer.from_file(str(VINDEX / 'tokenizer.json'))
    text = '<bos><|turn>user\n' + QUESTION + '<turn|>\n<|turn>model\n'
    ids = tok.encode(text, add_special_tokens=False).ids
    donor_ids = tok.encode(text.replace('hydrogen', 'helium'), add_special_tokens=False).ids
    assert len(ids) == len(donor_ids) == 30
    assert [(i,a,b) for i,(a,b) in enumerate(zip(ids,donor_ids)) if a!=b] == [(5,13076,42404)]
    write(root / 'session.json', dict(start=time.time(), source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=repo, text=True).strip(), binary=binary,
        binary_sha256=old['binary_sha256'], max_forwards=24, max_runtime_seconds=1800,
        question=QUESTION, prompt_ids=ids, donor_prompt_ids=donor_ids,
        source_entity_position=5, gains=[1], group=group_name, source_protocol=str(prior), max_prefix_proposals=12,
        scope='integrated from-empty-answer replay on observed failing paraphrase; fixed selected row group and gain 1; per-question contrast; no weight edit'))
    child = subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,str(VINDEX),'24'])
    deadline = time.monotonic()+1740
    count = 0
    summary = dict(status='running',prefix=[],candidates=[],baseline_tail=[],edited_tail=[])

    def run(label, prompt, generated, spec=None):
        nonlocal count
        assert count < 24
        name = f'{count:03}-{label}'
        job = dict(prompt_ids=prompt,seed_ids=list(generated),max_new_tokens=1)
        if spec is not None:
            job['intervention']=spec
        write(root/'jobs'/f'{name}.pending',job)
        (root/'jobs'/f'{name}.pending').rename(root/'jobs'/f'{name}.json')
        path = root/'results'/f'{name}.json'
        while not path.exists():
            if child.poll() is not None:
                raise RuntimeError(f'runner exited {child.returncode}')
            if time.monotonic()>deadline:
                raise TimeoutError('entity-first deadline')
            time.sleep(1)
        result=read(path);step=result['steps'][0]
        assert step['ordinal']==count
        count+=1
        arrays=dict(np.load(root/'results'/step['artifact']))
        assert all(np.isfinite(x).all() for x in arrays.values())
        assert int(arrays['logits'].argmax())==step['top1']
        print(name+': '+result['text'],flush=True)
        return step,arrays

    try:
        generated=[];boundary=None
        for _ in range(12):
            step,base=run('native-prefix',ids,generated)
            before=tok.decode(generated,skip_special_tokens=False)
            if step['text'].strip()=='H' and 'hydrogen' in before.lower():
                boundary=(step,base,list(generated))
                summary['boundary']=dict(job=step['job'],prefix=before,native_symbol=step['text'],native_symbol_id=step['top1'])
                break
            generated.append(step['top1'])
            summary['prefix'].append(dict(job=step['job'],token_id=step['top1'],text=step['text']))
            write(root/'progress.json',summary)
            if step['top1'] in [1,106]:
                break
        if boundary is None:
            summary.update(status='no eligible H boundary after naturally generated hydrogen within prefix budget',native_text=tok.decode(generated,skip_special_tokens=False))
        else:
            native,base,prefix=boundary
            donor_step,donor=run('donor-at-boundary',donor_ids,prefix)
            delta=donor['prefix_layer21']-base['prefix_layer21']
            assert np.all(delta[:5]==0)
            positions = dict(prompt=list(range(5,len(ids))), post_entity_prompt=list(range(6,len(ids))), answer=list(range(len(ids),len(ids)+len(prefix))), all_after_entity=list(range(5,len(ids)+len(prefix))))[group_name]
            summary['positions'] = positions
            np.savez_compressed(root/'boundary-contrast.npz',delta=delta,baseline=base['prefix_layer21'],donor=donor['prefix_layer21'])
            def spec(gain):
                return dict(layer=21,step=0,gain=gain,row_deltas=[dict(position=p,delta=delta[p].tolist()) for p in positions])
            zero,z=run('zero-gain',ids,prefix,spec(0))
            assert all(np.array_equal(z[k],v) for k,v in base.items())
            summary['identity_bit_equal']=True
            selected=None
            for gain in [1]:
                step,z=run(f'symbol-gain{gain}',ids,prefix,spec(gain))
                expected=base['prefix_layer21'].copy();expected[positions]+=np.float32(gain)*delta[positions]
                assert np.array_equal(z['prefix_layer21'],expected)
                trial=dict(job=step['job'],gain=gain,token=step['text'],token_id=step['top1'],exact_application=True)
                summary['candidates'].append(trial)
                write(root/'progress.json',summary)
                if step['text'].strip()=='He':
                    selected=trial;break
            if selected is None:
                summary['status']='no tested boundary gain produced He; stopped'
            else:
                summary['selected']=selected
                # Counterfactual baseline branch includes the original greedy H.
                baseline_ids=prefix+[native['top1']]
                baseline_eos=False
                for _ in range(min(4,max(0,24-count-4))):
                    step,_=run('baseline-tail',ids,baseline_ids)
                    baseline_ids.append(step['top1']);summary['baseline_tail'].append(step['job'])
                    if step['top1'] in [1,106]:
                        baseline_eos=True;break
                summary['baseline_text']=tok.decode(baseline_ids,skip_special_tokens=False)
                summary['baseline_eos']=baseline_eos
                # The chosen He is this intervention's greedy output, never forced.
                edited_ids=prefix+[selected['token_id']]
                edited_eos=False
                while count<24:
                    step,_=run('released-tail',ids,edited_ids)
                    edited_ids.append(step['top1']);summary['edited_tail'].append(step['job'])
                    summary['edited_text']=tok.decode(edited_ids,skip_special_tokens=False)
                    write(root/'progress.json',summary)
                    if step['top1'] in [1,106]:
                        edited_eos=True;break
                summary['edited_text']=tok.decode(edited_ids,skip_special_tokens=False)
                summary['edited_eos']=edited_eos
                summary['status']='entity-first paired branches tested' if edited_eos else 'edited branch truncated at fixed forward budget'
                summary['forced_output_tokens']=0
    except Exception as exc:
        summary.update(status='incomplete',error=repr(exc));raise
    finally:
        (root/'STOP').touch()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
        ep=root/'events.jsonl'
        events=[json.loads(x) for x in ep.read_text().splitlines()] if ep.exists() else []
        summary.update(attempted_forwards=sum(e['type']=='attempt' for e in events),completed_forwards=sum(e['type']=='complete' for e in events),child_returncode=child.returncode,end=time.time())
        write(root/'summary.json',summary)
    if child.returncode:
        raise RuntimeError(f'runner failed {child.returncode}')


if __name__=='__main__':
    main(*sys.argv[1:])
