"""Model-free H3 timing checker; same author as runner, not independent verification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SITES=(8,14,21)
ORDER=tuple(f'{q}_L{k:02}' for q in ('number','symbol') for k in SITES)+('lithium_L08','carbon_L08')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_call(root,name):
    r=next(x for x in read(root/'summary.json')['calls'] if x['name']==name)
    p=root/r['artifact'];assert p.parent==root and sha(p)==r['artifact_sha256']
    with np.load(p,allow_pickle=False) as z:return r,{k:z[k].copy() for k in z.files}


def assess(m):
    n,s=m['number'],m['symbol'];out={}
    for k in ('8','14'):
        a=n[k]['effect']-n['21']['effect'];b=s[k]['effect']-s['21']['effect']
        yes=bool(n[k]['effect']>=1 and a>=1 and n[k]['effect']>2*max(n[k]['random_effect'],0))
        out[k]=dict(number_specific_improvement=yes,number_early_minus_L21=a,symbol_early_minus_L21=b,
                    relative_contrast=a-b,timing_pattern=bool(yes and b<=1))
    order=bool(out['8']['timing_pattern'] and out['14']['timing_pattern'] and n['8']['effect']>=n['14']['effect']+1 and n['14']['effect']>=n['21']['effect']+1)
    if order:label='H3-compatible ordered pattern at these sites and common dose'
    elif any(x['timing_pattern'] for x in out.values()):label='Partial/local early-site compatibility; complete ordered pattern not met'
    elif any(x['number_specific_improvement'] for x in out.values()):label='Earlier number sensitivity without the relative symbol pattern'
    else:label='No H3 support at these sites and common dose; no general timing falsification'
    return dict(sites=out,ordered_pattern=order,reading=label)


def verify(root,source,tokenizer_path):
    s=read(root/'summary.json');session=read(root/'session.json');prior=read(source/'summary.json')
    assert s['synthetic'] is False and s['phase'] in ('h3-core','h3-released')
    assert s['status'] in ('complete','incomplete_released_cap')
    assert sha(source/'summary.json')==s['source_summary_sha256']
    assert read(source/'verification.json')['status']=='verified_complete'
    assert sha(tokenizer_path)==session['metadata_sha256']['tokenizer.json']
    tok=Tokenizer.from_file(str(tokenizer_path))
    assert len(s['calls'])==s['forwards']<=24 and len({r['name'] for r in s['calls']})==s['forwards']
    assert [r['ordinal'] for r in s['calls']]==list(range(s['forwards']))
    cg=s['cgroup']
    for k,n in [('memory.high',22),('memory.max',24),('memory.swap.max',8)]:assert int(cg[k])==n*2**30
    assert int(cg['memory.peak'])<=24*2**30 and int(cg['memory.swap.peak'])<=8*2**30
    recs={};data={}
    for r in s['calls']:
        rec,x=load_call(root,r['name']);assert rec==r
        assert all(np.isfinite(v).all() for v in x.values())
        assert int(x['logits'].argmax())==r['argmax'] and x['ple_before'].shape==(42,256)
        assert all(x[f'entity_L{k:02}'].shape==(2560,) for k in SITES)
        if r['patch_layer'] is not None:
            before=x['patch_before'].astype(np.float16);expected=before.copy();expected[5]+=x['requested_delta'].astype(np.float16)
            assert before.shape==(len(r['input_ids']),2560)
            assert np.array_equal(expected.astype(np.float32),x['patch_after'])
            assert np.array_equal(x['entity_L'+f"{r['patch_layer']:02}"],x['patch_before'][5])
        recs[r['name']],data[r['name']]=r,x
    v=dict(status='verified_complete',phase=s['phase'],forwards=s['forwards'],memory_limits_checked=True,
           verifier_author='same agent as runner; not independent verification')
    if s['phase']=='h3-released':
        cases=prior['release_cases'];assert 0<len(cases)<=8
        assert [b['name'] for b in s['branches']]==[c['name'] for c in cases]
        names=[];eos=set(session['eos_ids'])
        for c,b in zip(cases,s['branches']):
            _,x=load_call(source,c['name']);assert int(x['logits'].argmax())==c['first_token']
            emitted=[c['first_token']];assert len(b['ids'])==len(b['call_names'])+1<=4 and b['ids'][0]==emitted[0]
            for i,(name,t) in enumerate(zip(b['call_names'],b['ids'][1:])):
                r=recs[name];assert name==c['name']+f'_released{i}' and emitted[-1] not in eos
                assert r['input_ids']==c['prefix_ids']+emitted and r['argmax']==t and r['patch_layer'] is None
                names.append(name);emitted.append(t)
            marker=[105,4368,107];ids=c['prefix_ids'];starts=[i+3 for i in range(len(ids)-2) if ids[i:i+3]==marker];assert len(starts)==1
            assert tok.decode(ids[starts[0]:]+emitted,skip_special_tokens=False)==b['text']
            assert b['eos']==(emitted[-1] in eos)
            if not b['eos']:assert len(emitted)==4
        assert names==[r['name'] for r in s['calls']]
        complete=all(b['eos'] for b in s['branches'])
        assert s['status']==('complete' if complete else 'incomplete_released_cap')
        v.update(status='verified_complete' if complete else 'verified_incomplete',branches=s['branches']);return v
    assert s['forwards']==24 and s['sites']==list(SITES)
    assert sha(source/'verification.json')==s['source_verification_sha256']
    oldR=np.load(source/'residual.npy',allow_pickle=False);oldX=np.load(source/'random.npy',allow_pickle=False)
    N=float(np.linalg.norm(oldR.astype(np.float16).astype(np.float64)));assert N==s['common_norm']
    assert np.array_equal(np.load(root/'random.npy',allow_pickle=False),oldX)
    vectors={}
    for k in SITES:
        d=data['symbol_donor'][f'entity_L{k:02}']-data['symbol_base'][f'entity_L{k:02}']
        vector=oldR if k==21 else (d.astype(np.float64)/np.linalg.norm(d.astype(np.float64))*N).astype(np.float32)
        assert np.array_equal(vector,np.load(root/f'residual-L{k:02}.npy',allow_pickle=False))
        assert abs(np.linalg.norm(vector.astype(np.float16).astype(np.float64))/N-1)<=0.001
        vectors[k]=vector
    expected_names=set();metrics={};release=[];norms={}
    for q in ('symbol','number'):
        old,b=load_call(source,q+'_base');donor,d=load_call(source,q+'_donor')
        for mode,refrec,ref in [('base',old,b),('donor',donor,d)]:
            name=q+'_'+mode;expected_names.add(name)
            assert recs[name]['input_ids']==refrec['input_ids'] and recs[name]['patch_layer'] is None
            assert np.array_equal(data[name]['logits'],ref['logits']) and np.array_equal(data[name]['ple_before'],ref['ple_before'])
        for k in (SITES if q=='symbol' else (21,)):
            name=q+f'_zero_L{k:02}';expected_names.add(name);x=data[name]
            assert recs[name]['input_ids']==old['input_ids'] and recs[name]['patch_layer']==k
            assert np.count_nonzero(x['requested_delta'])==0 and np.array_equal(x['logits'],b['logits'])
        n=prior['metrics'][q]['native_id'];t=prior['metrics'][q]['target_id'];bm=float(b['logits'][t])-float(b['logits'][n]);metrics[q]={}
        for k in SITES:
            rn=q+f'_L{k:02}';xn=q+f'_random_L{k:02}';expected_names|={rn,xn}
            for name,vec in [(rn,vectors[k]),(xn,oldX)]:
                r=recs[name];x=data[name]
                assert r['patch_layer']==k and r['input_ids']==old['input_ids']
                assert np.array_equal(x['requested_delta'],vec.astype(np.float16).astype(np.float32))
                assert np.array_equal(x['ple_before'],b['ple_before'])
                assert np.array_equal(x['patch_before'][5],data[q+'_base'][f'entity_L{k:02}'])
            if k==21:
                for name,oldname in [(rn,q+'_residual'),(xn,q+'_random')]:
                    _,r=load_call(source,oldname);assert np.array_equal(data[name]['logits'],r['logits'])
            mr=float(data[rn]['logits'][t])-float(data[rn]['logits'][n]);mx=float(data[xn]['logits'][t])-float(data[xn]['logits'][n])
            metrics[q][str(k)]=dict(native_id=n,target_id=t,baseline_margin=bm,margin=mr,effect=mr-bm,random_effect=mx-bm,
                argmax=recs[rn]['argmax'],random_argmax=recs[xn]['argmax'],donor_argmax=recs[q+'_donor']['argmax'])
            assert metrics[q][str(k)]==s['metrics'][q][str(k)]
            if recs[rn]['argmax']!=n:
                r=recs[rn];release.append(dict(name=rn,question=q,prefix_ids=r['input_ids'],first_token=r['argmax'],native_id=n,target_id=t,
                    source_artifact=r['artifact'],source_artifact_sha256=r['artifact_sha256']))
        if q=='number':
            for k in SITES:
                for mode in ('base','donor'):
                    a=data['symbol_'+mode][f'entity_L{k:02}'].astype(np.float64);z=data['number_'+mode][f'entity_L{k:02}'].astype(np.float64)
                    rel=float(np.linalg.norm(a-z)/np.linalg.norm(a));assert rel<=0.001
                    assert s['causal_rows'][str(k)][mode]==dict(bit_identical=np.array_equal(a,z),relative_l2=rel)
    for off in s['off_target']:
        q=off['name'];old,b=load_call(source,q+'_base');bn=q+'_base';rn=q+'_L08';expected_names|={bn,rn}
        assert off['prefix_type']=='supplied' and recs[bn]['input_ids']==old['input_ids']==recs[rn]['input_ids']
        assert recs[bn]['patch_layer'] is None and recs[rn]['patch_layer']==8
        assert np.array_equal(data[bn]['logits'],b['logits']) and np.array_equal(data[rn]['ple_before'],b['ple_before'])
        assert np.array_equal(data[rn]['requested_delta'],vectors[8].astype(np.float16).astype(np.float32))
        assert off['native_argmax']==recs[bn]['argmax'] and off['residual_argmax']==recs[rn]['argmax']
        if off['native_argmax']!=off['residual_argmax']:
            r=recs[rn];release.append(dict(name=rn,question=q,prefix_ids=r['input_ids'],first_token=r['argmax'],native_id=off['native_argmax'],target_id=None,
                source_artifact=r['artifact'],source_artifact_sha256=r['artifact_sha256']))
    assert {o['name'] for o in s['off_target']}=={'lithium','carbon'} and set(recs)==expected_names
    for name in s['norms']:
        x=data[name];cast=float(np.linalg.norm(x['requested_delta'].astype(np.float64)))
        actual=float(np.linalg.norm(x['patch_after'][5].astype(np.float64)-x['patch_before'][5].astype(np.float64)))
        native=float(np.linalg.norm(x['patch_before'][5].astype(np.float64)))
        assert abs(cast/N-1)<=0.001 and abs(actual/N-1)<=0.01 and native>0
        norms[name]=dict(cast_norm=cast,actual_norm=actual,native_row_norm=native,relative_dose=actual/native)
        assert norms[name]==s['norms'][name]
    assert len(norms)==14
    expected=assess(metrics);assert expected==s['readings']
    release.sort(key=lambda x:ORDER.index(x['name']));assert release==s['release_cases']
    v.update(metrics=metrics,readings=expected,norms=norms,off_target=s['off_target'],release_cases=release);return v


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);a=p.parse_args();v=verify(a.root,a.source,a.tokenizer)
    (a.root/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
