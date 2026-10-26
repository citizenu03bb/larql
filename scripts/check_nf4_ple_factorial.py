"""Model-free PLE factorial evidence checker, same author as the runner.

Checks source replays, exact intervention arrays, random row norms, factorial
margins/endpoints, release ancestry and cgroup bounds. Does not infer efficacy
from the verification status and does not load model weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

MODES=('base','residual','sham','ple','both','random_ple','residual_random_ple','donor')
ORDER=('number_both','number_ple','symbol_both','symbol_ple')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def call(root,name):
    rec=next(r for r in read(root/'summary.json')['calls'] if r['name']==name)
    p=root/rec['artifact'];assert p.parent==root and sha(p)==rec['artifact_sha256']
    with np.load(p,allow_pickle=False) as z:return rec,{k:z[k].copy() for k in z.files}


def random_expected(base,donor):
    target=np.linalg.norm(donor.astype(np.float64)-base.astype(np.float64),axis=1)
    x=np.random.default_rng(20261027).standard_normal(base.shape)
    requested=(x/np.linalg.norm(x,axis=1)[:,None]*target[:,None]).astype(np.float32)
    after=(base.astype(np.float16)+requested.astype(np.float16)).astype(np.float32)
    actual=np.linalg.norm(after.astype(np.float64)-base.astype(np.float64),axis=1)
    errors=np.zeros_like(target);nonzero=target>0
    errors[nonzero]=np.abs(actual[nonzero]/target[nonzero]-1)
    assert np.all(errors<=0.001) and np.all(actual[~nonzero]==0)
    return requested,after,dict(target_norms=target.tolist(),actual_norms=actual.tolist(),relative_errors=errors.tolist())


def passed(effect,random):return bool(effect>=1.0 and effect>2*max(random,0.0))


def verify(root,source,tokenizer_path):
    s=read(root/'summary.json');session=read(root/'session.json')
    assert not s['synthetic'] and s['status'] in ('complete','incomplete_released_cap')
    assert sha(tokenizer_path)==session['metadata_sha256']['tokenizer.json']
    tok=Tokenizer.from_file(str(tokenizer_path))
    phase=s['phase'];is_release=phase=='ple-released';assert phase in ('ple-factorial','ple-released')
    assert s['forwards']==len(s['calls'])<=(16 if is_release else 24)
    assert [r['ordinal'] for r in s['calls']]==list(range(s['forwards']))
    assert len({r['name'] for r in s['calls']})==s['forwards']
    source_summary=read(source/'summary.json')
    assert sha(source/'summary.json')==s['source_summary_sha256']
    assert read(source/'verification.json')['status']=='verified_complete'
    cg=s['cgroup']
    for key,n in [('memory.high',22),('memory.max',24),('memory.swap.max',8)]:assert int(cg[key])==n*2**30
    assert int(cg['memory.peak'])<=24*2**30 and int(cg['memory.swap.peak'])<=8*2**30
    records={};arrays={}
    for rec in s['calls']:
        r,x=call(root,rec['name']);assert r==rec
        assert all(np.isfinite(v).all() for v in x.values())
        assert int(x['logits'].argmax())==rec['argmax'] and x['ple_before'].shape==(42,256)
        if rec['residual']:
            expected=x['l21_before'].astype(np.float16).copy()
            expected[5]+=x['requested_residual'].astype(np.float16)
            assert np.array_equal(expected.astype(np.float32),x['l21_after'])
        records[rec['name']],arrays[rec['name']]=rec,x
    result=dict(status='verified_complete',phase=phase,forwards=s['forwards'],
                verifier_author='same agent as runner; not independent verification',memory_limits_checked=True)
    if is_release:
        expected=source_summary['release_cases'];assert 0<len(expected)<=4
        assert [b['name'] for b in s['branches']]==[c['name'] for c in expected]
        all_names=[];eos=set(session['eos_ids'])
        for case,branch in zip(expected,s['branches']):
            _,x=call(source,case['name'])
            assert int(x['logits'].argmax())==case['first_token']
            assert branch['ids'][0]==case['first_token']
            assert len(branch['ids'])==len(branch['call_names'])+1<=5
            emitted=[case['first_token']]
            for i,(name,t) in enumerate(zip(branch['call_names'],branch['ids'][1:])):
                rec=records[name];all_names.append(name)
                assert name==case['name']+f'_released{i}'
                assert emitted[-1] not in eos and rec['input_ids']==case['prefix_ids']+emitted
                assert rec['argmax']==t and not any(rec[k] for k in ('residual','ple_replacement','ple_addition'))
                emitted.append(t)
            ids=case['prefix_ids'];marker=[105,4368,107]
            starts=[i+3 for i in range(len(ids)-2) if ids[i:i+3]==marker];assert len(starts)==1
            assert tok.decode(ids[starts[0]:]+emitted,skip_special_tokens=False)==branch['text']
            assert branch['target_first']==(emitted[0]==case['target_id'])
            assert branch['eos']==(emitted[-1] in eos)
            if not branch['eos']:assert len(emitted)==5
        assert all_names==[r['name'] for r in s['calls']]
        complete=all(b['eos'] for b in s['branches'])
        assert s['status']==('complete' if complete else 'incomplete_released_cap')
        result.update(branches=s['branches'],status='verified_complete' if complete else 'verified_incomplete')
        return result
    assert s['forwards']==22 and s['modes']==list(MODES)
    assert source_summary['core_gate'] is False
    assert sha(source/'verification.json')==s['source_verification_sha256']
    assert sha(source/'residual.npy')==s['residual_sha256']==sha(root/'residual.npy')
    delta=np.load(root/'residual.npy',allow_pickle=False)
    metrics={};release=[]
    for q in ('symbol','number','lithium','carbon'):
        old,b=call(source,q+'_base');_,d=call(source,('symbol' if q in ('lithium','carbon') else q)+'_donor')
        expected_random,random_after,stats=random_expected(b['ple_before'],d['ple_before'])
        assert np.array_equal(np.load(root/(q+'-random-ple.npy'),allow_pickle=False),expected_random)
        assert s['random_norms'][q]==stats
        modes=MODES if q in ('symbol','number') else ('base','both','residual_random_ple')
        for mode in modes:
            rec=records[q+'_'+mode];x=arrays[q+'_'+mode]
            source_rec,source_data=call(source,q+'_'+mode) if mode in ('base','residual','donor') else (old,b)
            assert rec['input_ids']==source_rec['input_ids']
            r=mode in ('residual','both','residual_random_ple')
            p=mode in ('sham','ple','both');a=mode in ('random_ple','residual_random_ple')
            assert (rec['residual'],rec['ple_replacement'],rec['ple_addition'])==(r,p,a)
            assert np.array_equal(x['ple_before'],d['ple_before'] if mode=='donor' else b['ple_before'])
            if r:assert np.array_equal(x['requested_residual'],delta.astype(np.float16).astype(np.float32))
            if p:assert np.array_equal(x['ple_after'],b['ple_before'] if mode=='sham' else d['ple_before'])
            if a:assert np.array_equal(x['ple_after'],random_after)
            if mode in ('base','residual','donor','sham'):
                assert np.array_equal(x['logits'],source_data['logits'])
        if q not in ('symbol','number'):
            off=next(o for o in s['off_target'] if o['name']==q)
            assert off['prefix_type']=='supplied'
            assert off['argmax']=={m:records[q+'_'+m]['argmax'] for m in modes}
            continue
        native=source_summary['metrics'][q]['native_id'];target=source_summary['metrics'][q]['target_id']
        m={mode:float(arrays[q+'_'+mode]['logits'][target])-float(arrays[q+'_'+mode]['logits'][native]) for mode in MODES}
        ep=m['ple']-m['base'];er=m['both']-m['residual'];rp=m['random_ple']-m['base'];rr=m['residual_random_ple']-m['residual']
        computed=dict(native_id=native,target_id=target,margins=m,
            argmax={mode:records[q+'_'+mode]['argmax'] for mode in MODES},ple_effect=ep,conditional_ple_effect=er,
            random_ple_effect=rp,conditional_random_ple_effect=rr,interaction=m['both']-m['residual']-m['ple']+m['base'],
            ple_endpoint_passed=passed(ep,rp),conditional_endpoint_passed=passed(er,rr))
        assert computed==s['metrics'][q];metrics[q]=computed
        for mode in ('ple','both'):
            r=records[q+'_'+mode]
            if r['argmax']!=native:
                release.append(dict(name=q+'_'+mode,question=q,source_artifact=r['artifact'],source_artifact_sha256=r['artifact_sha256'],
                                    prefix_ids=r['input_ids'],first_token=r['argmax'],target_id=target,native_id=native))
    assert set(records)=={q+'_'+m for q in ('symbol','number') for m in MODES}|{q+'_'+m for q in ('lithium','carbon') for m in ('base','both','residual_random_ple')}
    release.sort(key=lambda x:ORDER.index(x['name']));assert release==s['release_cases']
    ep=metrics['number']['ple_endpoint_passed'];er=metrics['number']['conditional_endpoint_passed']
    if ep and er:reading='PLE direction contributes with and without R on this number boundary'
    elif ep:reading='PLE contribution without R; conditional-on-R criterion not met'
    elif er:reading='PLE contribution contingent on R; PLE-alone criterion not met'
    else:reading='Neither frozen effect-size/direction-control criterion met; no general PLE falsification'
    assert reading==s['interpretation']
    result.update(metrics=metrics,off_target=s['off_target'],release_cases=release,interpretation=s['interpretation'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);a=p.parse_args()
    v=verify(a.root,a.source,a.tokenizer)
    (a.root/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
