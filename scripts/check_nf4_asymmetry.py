"""Check NF4 asymmetry arrays/ancestry without model inference; same author as runner."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def margin(x,target,native):return float(x[target])-float(x[native])


def verify(root,tokenizer_path,bench=None,native_root=None):
    s=read(root/'summary.json');session=read(root/'session.json')
    assert not s['synthetic'] and s['status'] in ('complete','incomplete_native_cap')
    assert sha(tokenizer_path)==session['metadata_sha256']['tokenizer.json']
    tok=Tokenizer.from_file(str(tokenizer_path))
    def token(t):
        x=tok.encode(t,add_special_tokens=False).ids;assert len(x)==1;return x[0]
    limit=16 if s['phase']=='eligibility' else 24
    assert len(s['calls'])==s['forwards']<=limit
    assert [r['ordinal'] for r in s['calls']]==list(range(s['forwards']))
    records={};data={}
    for r in s['calls']:
        p=root/r['artifact'];assert p.parent==root and sha(p)==r['artifact_sha256']
        with np.load(p,allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
        assert all(np.isfinite(a).all() for a in x.values())
        assert int(x['logits'].argmax())==r['argmax']
        assert x['ple_before'].shape==(42,256)
        assert not r['ple_replacement'] and not r['ple_addition']
        if r['residual']:
            before=x['l21_before'].astype(np.float16);after=x['l21_after']
            expected=before.copy();expected[5]+=x['requested_residual'].astype(np.float16)
            assert np.array_equal(expected.astype(np.float32),after)
        records[r['name']]=r;data[r['name']]=x
    cg=s['cgroup']
    for key,n in [('memory.high',22),('memory.max',24),('memory.swap.max',8)]:assert int(cg[key])==n*2**30
    assert int(cg['memory.peak'])<=24*2**30 and int(cg['memory.swap.peak'])<=8*2**30
    result=dict(status='verified_complete',phase=s['phase'],forwards=s['forwards'],
        verifier_author='same agent as runner; not independent verification',memory_limits_checked=True)
    eos=set(session['eos_ids'])
    if s['phase']=='eligibility':
        n=s['native'];prefix=[]
        for i,(name,t) in enumerate(zip(n['call_names'],n['ids'])):
            r=records[name]
            assert r['name']==f'number_native{i:02}' and r['input_ids']==n['prompt_ids']+prefix
            assert r['argmax']==t and not r['residual']
            assert not prefix or prefix[-1] not in eos
            prefix.append(t)
        assert len(prefix)==s['forwards']==len(n['call_names'])
        assert tok.decode(prefix,skip_special_tokens=False)==n['text']
        assert n['eos']==(prefix[-1] in eos)
        digits=[name for name,t in zip(n['call_names'],prefix) if tok.decode([t],skip_special_tokens=False).strip() in set('0123456789')]
        boundary=digits[0] if digits else None
        assert boundary==n['boundary']
        good=bool(n['eos'] and boundary and records[boundary]['argmax']==token('1') and 'hydrogen' in n['text'].lower())
        assert s['eligible']==good
        result.update(eligible=good,native_text=n['text'],boundary=boundary)
        if not n['eos']:result['status']='verified_incomplete'
    else:
        assert bench and native_root
        ns=read(native_root/'summary.json');bs=read(bench/'summary.json')
        assert ns['eligible'] and read(native_root/'verification.json')['status']=='verified_complete'
        assert sha(bench/'summary.json')==s['source_symbol_summary_sha256']
        assert sha(native_root/'summary.json')==s['source_number_summary_sha256']
        prompt=read(bench/'session.json')['prompt_ids']
        source_names={
            'symbol':next(name for name,t in zip(bs['native']['call_names'],bs['native']['ids']) if t==token('H')),
            'number':ns['native']['boundary']}
        for q,src in [('symbol',bench),('number',native_root)]:
            raw_summary=bs if q=='symbol' else ns
            source=next(c for c in raw_summary['calls'] if c['name']==source_names[q])
            assert records[q+'_base']['input_ids']==source['input_ids']
            assert sha(src/source['artifact'])==source['artifact_sha256']
            with np.load(src/source['artifact'],allow_pickle=False) as z:
                assert np.array_equal(data[q+'_base']['logits'],z['logits'].reshape(-1))
        delta=np.load(root/'residual.npy',allow_pickle=False);random=np.load(root/'random.npy',allow_pickle=False)
        raw=data['symbol_donor']['l21_before']-data['symbol_base']['l21_before']
        gain=np.float32(np.linalg.norm(raw[5:len(prompt)].astype(np.float64))/np.linalg.norm(raw[5].astype(np.float64)))
        assert np.array_equal(delta,(gain*raw[5]).astype(np.float32))
        norm=np.linalg.norm(delta.astype(np.float16).astype(np.float64))
        draw=np.random.default_rng(20261026).standard_normal(delta.shape)
        assert np.array_equal(random,(draw/np.linalg.norm(draw)*norm).astype(np.float32))
        assert abs(np.linalg.norm(random.astype(np.float16).astype(np.float64))/norm-1)<=0.001
        assert s['norm_rule']['gain']==float(gain) and s['norm_rule']['actual_norm']==norm
        metrics={}
        for q,baseid,target in [('symbol',token('H'),token('He')),('number',token('1'),token('2'))]:
            base=data[q+'_base'];ids=records[q+'_base']['input_ids']
            donor=list(ids);donor[5]=token(' helium')
            assert records[q+'_donor']['input_ids']==donor
            for mode in ('zero','residual','random'):
                assert records[q+'_'+mode]['input_ids']==ids
                expected=np.zeros_like(delta) if mode=='zero' else delta if mode=='residual' else random
                assert np.array_equal(data[q+'_'+mode]['requested_residual'],expected.astype(np.float16).astype(np.float32))
                assert np.array_equal(base['l21_before'],data[q+'_'+mode]['l21_before'])
                assert np.array_equal(base['ple_before'],data[q+'_'+mode]['ple_before'])
            assert np.array_equal(base['logits'],data[q+'_zero']['logits'])
            assert records[q+'_base']['argmax']==baseid
            bm=margin(base['logits'],target,baseid)
            m=dict(native_id=baseid,target_id=target,baseline_margin=bm,
                residual_margin=margin(data[q+'_residual']['logits'],target,baseid),
                residual_increment=margin(data[q+'_residual']['logits'],target,baseid)-bm,
                random_increment=margin(data[q+'_random']['logits'],target,baseid)-bm,
                residual_argmax=records[q+'_residual']['argmax'],donor_argmax=records[q+'_donor']['argmax'])
            assert m==s['metrics'][q];metrics[q]=m
        for label in ('base','donor'):
            x=data['symbol_'+label]['l21_before'][5].astype(np.float64)
            y=data['number_'+label]['l21_before'][5].astype(np.float64)
            rel=float(np.linalg.norm(x-y)/np.linalg.norm(x))
            assert rel<=0.001 and s['causal_row_check'][label]==dict(bit_identical=np.array_equal(x,y),relative_l2=rel)
        assert len(s['off_target'])==2
        for off in s['off_target']:
            name=off['name'];ids=off['prompt_ids']+off['supplied_seed_ids']
            for mode,v in [('base',None),('residual',delta),('random',random)]:
                r=records[name+'_'+mode];assert r['input_ids']==ids
                assert r['argmax']==off[{'base':'baseline_argmax','residual':'residual_argmax','random':'random_argmax'}[mode]]
                if v is not None:assert np.array_equal(data[name+'_'+mode]['requested_residual'],v.astype(np.float16).astype(np.float32))
        sm,num=metrics['symbol'],metrics['number']
        gate=bool(sm['donor_argmax']==sm['target_id'] and num['donor_argmax']==num['target_id']
            and sm['residual_argmax']==sm['target_id'] and num['residual_argmax']==num['native_id']
            and sm['residual_increment']>0 and sm['residual_increment']>2*max(sm['random_increment'],0)
            and sm['residual_increment']>2*max(num['residual_increment'],0))
        assert gate==s['core_gate']
        assert len(s['continuations'])==(2 if gate else 0)
        for name,c in s['continuations'].items():
            initial=records[name+'_residual'];prefix=initial['input_ids'];emitted=[]
            for i,(call,t) in enumerate(zip(c['call_names'],c['ids'])):
                r=records[call];assert r['input_ids']==prefix+emitted and r['argmax']==t
                if i:assert not r['residual']
                emitted.append(t)
            p=prompt if name=='symbol' else ns['native']['prompt_ids']
            assert tok.decode(prefix[len(p):]+emitted,skip_special_tokens=False)==c['text']
            assert c['eos']==(emitted[-1] in eos)
        ready=bool(gate and all(c['eos'] for c in s['continuations'].values())
                   and 'helium' not in s['continuations']['symbol']['text'].lower())
        assert ready==s['transfer_reproduced']
        result.update(transfer_reproduced=ready,metrics=metrics,continuations=s['continuations'],off_target=s['off_target'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--bench-source',type=Path);p.add_argument('--native-source',type=Path)
    a=p.parse_args();v=verify(a.root,a.tokenizer,a.bench_source,a.native_source)
    (a.root/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
