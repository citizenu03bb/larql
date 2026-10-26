"""Frozen PLE factorial protocol, 2026-10-26, before any PLE model intervention.

Mandate: a new margin-based causal test authorized after the failed top-1 gate
in nf4-asymmetry-20261026. Preserve that negative gate. No new residual gain,
question, boundary, donor, layer search or free-form H->He tuning. Purpose:
turn the observed margin asymmetry into an auditable mechanistic result.

R: EXACT stored residual.npy, applied once post-L21 at entity row 5. P: replace
ONLY the FINAL PROJECTED per-layer input at row 5 with the archived complete
helium donor's row, at all 42 decoder blocks. Leave input ids/main embedding
native. P includes both token-lookup and projected-embedding contributions;
it does not isolate the lookup table alone. R+P is a factorial intervention,
not a decomposition of the full donor or an exact entity transplant.

Same two saved native value boundaries: symbol H/He in LaTeX; number one/two.
For EACH: native, R, self-P sham, P, R+P, random-P, R+random-P, full donor
(16 forwards). Native/R/donor must reproduce archived full logits bit for bit;
self-P must equal native. Use the archived R random controls as context, not
new evidence. Two off-target supplied-prefix questions (lithium/carbon symbol):
native, R+P, R+random-P (6), for 22 planned forwards, hard service cap 24.
They test local leakage only, not free-chat preservation or general selectivity.

Random-P: one Gaussian draw seed 20261027, rows independently scaled to the
L2 norm of donor-minus-native PLE at EACH layer. Cast to float16 once, no search
or iterative correction. Require actual rounded addition norms within 0.1%
of the transplant change at every nonzero layer; otherwise instrument failure.
Construct separately from each question's stored PLE rows by this same rule.

Read m00(native), m10(R), m01(P), m11(R+P), each target-minus-native margin.
Primary literal H2 endpoint: EP=m01-m00 on NUMBER. Conditional endpoint for the
old R setting: EPR=m11-m10 on NUMBER. Controls are their respective matched
random-P increments. Each endpoint passes if >=1 logit AND >2*max(random,0).
This is a practical one-case threshold, NOT a significance test. Report all
margins/argmaxes, EP, EPR and interaction m11-m10-m01+m00 for both attributes.

Frozen readings: both pass -> PLE input direction contributes with/without R
on this number boundary; primary only -> causal contribution without R, not a
rescue under R; conditional only -> contribution contingent on R; neither ->
no support at this predeclared effect size/direction control, not zero effect
or general falsification of PLE. None identifies the unique cause of asymmetry,
excludes name-form H1, demonstrates selectivity, persistence or knowledge editing.
Top-1 is secondary and never gates the margin reading.

Released phase, conditional ONLY on changed argmax in the FOUR fixed primary
P/R+P cases: number R+P, number P, symbol R+P, symbol P, in that order. Reuse
that actually selected first token, no forcing, then remove all interventions
and generate <=4 more tokens or EOS per case. <=16 forwards, hard cap 16 in a
second sequential service. Inspect every changed primary case, including an
unexpected formatting token. Truncations stay incomplete. These are resumed
native-prefix continuations, NOT an integrated edited run from empty.

Total <=38 new forwards, two sequential services (22 + <=16), one E4B worker,
22G/24G/8G and RuntimeMaxSec=1800 each, runner deadline 1740 s. Stop on cap,
deadline, OOM, nonfinite data, failed source/sham replay, failed norm or hook
checks. No expansion after results. If no primary argmax changes, skip release.
Archive hashes, model-free same-author verifier, report/journal/status/commit.
No weight/vindex/persistent-patch changes or public transmission.
"""
import argparse
import json
from pathlib import Path
import traceback
import tempfile
from types import SimpleNamespace

import numpy as np

from run_nf4_asymmetry import Harness, saved_call, margin, dry_run as hook_dry_run
from run_nf4_local_bench import sha, write

PLE_SEED=20261027
MODES=('base','residual','sham','ple','both','random_ple','residual_random_ple','donor')
RELEASE_ORDER=('number_both','number_ple','symbol_both','symbol_ple')


def random_ple(base, donor):
    change=donor.astype(np.float64)-base.astype(np.float64)
    norms=np.linalg.norm(change,axis=1)
    draw=np.random.default_rng(PLE_SEED).standard_normal(change.shape)
    scaled=(draw/np.linalg.norm(draw,axis=1)[:,None]*norms[:,None]).astype(np.float32)
    actual=(base.astype(np.float16)+scaled.astype(np.float16)).astype(np.float64)-base.astype(np.float64)
    got=np.linalg.norm(actual,axis=1)
    nonzero=norms>0
    errors=np.zeros_like(norms);errors[nonzero]=np.abs(got[nonzero]/norms[nonzero]-1)
    assert np.all(errors<=0.001), 'rounded random-P layer norm mismatch'
    assert np.all(got[~nonzero]==0)
    return scaled,dict(target_norms=norms.tolist(),actual_norms=got.tolist(),relative_errors=errors.tolist())


def read(p):return json.loads(p.read_text())


def endpoint(effect, random):
    return bool(effect>=1.0 and effect>2*max(random,0.0))


def interpret(metrics):
    a=metrics['number']['ple_endpoint_passed'];b=metrics['number']['conditional_endpoint_passed']
    if a and b:return 'PLE direction contributes with and without R on this number boundary'
    if a:return 'PLE contribution without R; conditional-on-R criterion not met'
    if b:return 'PLE contribution contingent on R; PLE-alone criterion not met'
    return 'Neither frozen effect-size/direction-control criterion met; no general PLE falsification'


def factorial(h,source):
    prior=read(source/'summary.json');v=read(source/'verification.json')
    assert prior['synthetic']==h.summary['synthetic']
    assert v['status']=='verified_complete' and prior['status']=='complete'
    assert prior['core_gate'] is False and prior['transfer_reproduced'] is False
    assert prior['metrics']['symbol']['residual_increment']>2*max(prior['metrics']['number']['residual_increment'],0)
    delta=np.load(source/'residual.npy',allow_pickle=False)
    np.save(h.root/'residual.npy',delta)
    h.summary.update(source_summary_sha256=sha(source/'summary.json'),
        source_verification_sha256=sha(source/'verification.json'),residual_sha256=sha(source/'residual.npy'),
        random_seed=PLE_SEED,modes=list(MODES),metrics={},off_target=[],release_cases=[])
    for q in ('symbol','number'):
        old_base,b=saved_call(source,q+'_base');old_donor,d=saved_call(source,q+'_donor')
        _,old_r=saved_call(source,q+'_residual')
        baseline=b['ple_before'];donor=d['ple_before'];random,spec=random_ple(baseline,donor)
        np.save(h.root/(q+'-random-ple.npy'),random)
        h.summary.setdefault('random_norms',{})[q]=spec
        cases={}
        for mode in MODES:
            ids=old_donor['input_ids'] if mode=='donor' else old_base['input_ids']
            r=delta if mode in ('residual','both','residual_random_ple') else None
            p=donor if mode in ('ple','both') else baseline if mode=='sham' else None
            a=random if mode in ('random_ple','residual_random_ple') else None
            rec,data=h.forward(q+'_'+mode,ids,residual=r,ple_replacement=p,ple_addition=a)
            assert np.array_equal(data['ple_before'],d['ple_before'] if mode=='donor' else baseline)
            if mode in ('base','donor','residual','sham'):
                expected=b if mode in ('base','sham') else d if mode=='donor' else old_r
                assert np.array_equal(data['logits'],expected['logits']), 'source/sham replay differs'
            cases[mode]=(rec,data)
        native=prior['metrics'][q]['native_id'];target=prior['metrics'][q]['target_id']
        margins={mode:margin(data,target,native) for mode,(_,data) in cases.items()}
        ep=margins['ple']-margins['base'];er=margins['both']-margins['residual']
        rp=margins['random_ple']-margins['base'];rr=margins['residual_random_ple']-margins['residual']
        h.summary['metrics'][q]=dict(native_id=native,target_id=target,margins=margins,
            argmax={m:rec['argmax'] for m,(rec,_) in cases.items()},ple_effect=ep,conditional_ple_effect=er,
            random_ple_effect=rp,conditional_random_ple_effect=rr,
            interaction=margins['both']-margins['residual']-margins['ple']+margins['base'],
            ple_endpoint_passed=endpoint(ep,rp),conditional_endpoint_passed=endpoint(er,rr))
        for mode in ('ple','both'):
            rec,_=cases[mode]
            if rec['argmax']!=native:
                h.summary['release_cases'].append(dict(name=q+'_'+mode,question=q,
                    source_artifact=rec['artifact'],source_artifact_sha256=rec['artifact_sha256'],
                    prefix_ids=rec['input_ids'],first_token=rec['argmax'],target_id=target,native_id=native))
    for off in prior['off_target']:
        name=off['name'];old,b=saved_call(source,name+'_base');_,d=saved_call(source,'symbol_donor')
        donor=d['ple_before'];random,spec=random_ple(b['ple_before'],donor)
        np.save(h.root/(name+'-random-ple.npy'),random)
        h.summary['random_norms'][name]=spec
        cases={}
        for mode in ('base','both','residual_random_ple'):
            rec,data=h.forward(name+'_'+mode,old['input_ids'],
                residual=delta if mode!='base' else None,
                ple_replacement=donor if mode=='both' else None,
                ple_addition=random if mode=='residual_random_ple' else None)
            if mode=='base':assert np.array_equal(data['logits'],b['logits'])
            cases[mode]=rec['argmax']
        h.summary['off_target'].append(dict(name=name,prefix_type='supplied',argmax=cases))
    h.summary['release_cases'].sort(key=lambda x:RELEASE_ORDER.index(x['name']))
    h.summary.update(status='complete',interpretation=interpret(h.summary['metrics']))


def released(h,source):
    s=read(source/'summary.json');v=read(source/'verification.json')
    assert s['synthetic']==h.summary['synthetic']
    assert s['status']=='complete' and v['status']=='verified_complete'
    assert s['release_cases'] and len(s['release_cases'])<=4
    h.summary.update(source_summary_sha256=sha(source/'summary.json'),branches=[])
    for case in s['release_cases']:
        assert sha(source/case['source_artifact'])==case['source_artifact_sha256']
        emitted=[case['first_token']];calls=[]
        for i in range(4):
            if emitted[-1] in h.eos:break
            rec,_=h.forward(case['name']+f'_released{i}',case['prefix_ids']+emitted)
            emitted.append(rec['argmax']);calls.append(rec['name'])
        # Chat assistant start is an existing fixed suffix in the saved native ids.
        ids=case['prefix_ids'];marker=[105,4368,107]
        starts=[i+3 for i in range(len(ids)-2) if ids[i:i+3]==marker]
        assert len(starts)==1
        text=h.decode(ids[starts[0]:]+emitted)
        h.summary['branches'].append(dict(name=case['name'],ids=emitted,call_names=calls,
            text=text,eos=emitted[-1] in h.eos,target_first=emitted[0]==case['target_id']))
    h.summary['status']='complete' if all(b['eos'] for b in h.summary['branches']) else 'incomplete_released_cap'


def dry_run():
    hook_dry_run()
    b=np.arange(42*256,dtype=np.float32).reshape(42,256)/4096
    b=b.astype(np.float16).astype(np.float32);d=(b+0.5).astype(np.float16).astype(np.float32)
    r,n=random_ple(b,d);assert r.shape==(42,256) and max(n['relative_errors'])<=0.001
    assert endpoint(1.0,0) and not endpoint(0.99,0) and not endpoint(2.0,1.0)
    assert not endpoint(-1.0,-5)
    synthetic_orchestration()
    print('SYNTHETIC factorial dry-run: frozen norm rule and endpoint edge cases pass; no model evidence')


def synthetic_orchestration():
    """Run the actual 22-condition scheduler and release logic with synthetic arrays."""
    root=Path(tempfile.mkdtemp(prefix='larql-ple-factorial-synthetic-',dir='/tmp'))
    b=np.zeros((42,256),np.float32);d=np.ones_like(b)
    delta=np.ones(256,np.float32)*0.125
    def fake_arrays(q,residual=None,replacement=None,addition=None,donor=False,ending=False):
        native,target=(0,1) if q==0 else (2,3) if q==2 else (4,1) if q==4 else (5,1)
        logits=np.zeros(16,np.float32);logits[native]=10
        logits[target]=(11 if donor else 0)+(2 if residual is not None else 0)
        if replacement is not None and np.any(replacement):logits[target]+=15
        if addition is not None:logits[target]+=0.01
        if ending:logits[:]=0;logits[9]=20
        before=np.zeros((13,256),np.float32)
        x=dict(logits=logits,ple_before=d.copy() if donor else b.copy(),l21_before=before)
        if replacement is not None:x['ple_after']=replacement.copy()
        if addition is not None:x['ple_after']=(b.astype(np.float16)+addition.astype(np.float16)).astype(np.float32)
        if residual is not None:
            x['requested_residual']=residual.astype(np.float16).astype(np.float32)
            x['l21_after']=before.copy();x['l21_after'][5]=x['requested_residual']
        return x
    class Fake:
        def __init__(self,path,never_end=False):
            self.root=path;path.mkdir();self.eos={9};self.never_end=never_end
            self.summary=dict(status='running',synthetic=True,calls=[],forwards=0)
            self.limit=SimpleNamespace(count=0)
        def forward(self,name,ids,residual=None,ple_replacement=None,ple_addition=None):
            end=len(ids)>13 and not self.never_end
            x=fake_arrays(ids[4],residual,ple_replacement,ple_addition,ids[5]==99,end)
            filename=f'{self.limit.count:03}-{name}.npz';np.savez_compressed(self.root/filename,**x)
            rec=dict(name=name,input_ids=ids.copy(),artifact=filename,artifact_sha256=sha(self.root/filename),argmax=int(x['logits'].argmax()))
            self.summary['calls'].append(rec);self.limit.count+=1;self.summary['forwards']=self.limit.count
            return rec,x
        def decode(self,ids):return ' '.join(map(str,ids))
    source=root/'source';source.mkdir();calls=[]
    for name,q in [('symbol',0),('number',2),('lithium',4),('carbon',5)]:
        for mode in (('base','donor','residual') if q in (0,2) else ('base',)):
            ids=[2,105,2364,107,q,99 if mode=='donor' else 10,1,106,107,105,4368,107,42]
            x=fake_arrays(q,delta if mode=='residual' else None,donor=mode=='donor')
            file=name+'_'+mode+'.npz';np.savez_compressed(source/file,**x)
            calls.append(dict(name=name+'_'+mode,input_ids=ids,artifact=file,artifact_sha256=sha(source/file)))
    write(source/'summary.json',dict(status='complete',synthetic=True,core_gate=False,transfer_reproduced=False,calls=calls,
        metrics={'symbol':{'residual_increment':2,'native_id':0,'target_id':1},'number':{'residual_increment':0,'native_id':2,'target_id':3}},
        off_target=[{'name':'lithium'},{'name':'carbon'}]))
    write(source/'verification.json',{'status':'verified_complete','synthetic':True});np.save(source/'residual.npy',delta)
    h=Fake(root/'factorial');factorial(h,source)
    assert h.limit.count==22 and len(h.summary['release_cases'])==4
    assert h.summary['metrics']['number']['ple_endpoint_passed']
    write(h.root/'summary.json',h.summary);write(h.root/'verification.json',{'status':'verified_complete','synthetic':True})
    f=Fake(root/'released');released(f,h.root)
    assert f.limit.count==4 and f.summary['status']=='complete'
    cap=Fake(root/'released-cap',never_end=True);released(cap,h.root)
    assert cap.limit.count==16 and cap.summary['status']=='incomplete_released_cap'
    for fake in (f,cap):write(fake.root/'summary.json',fake.summary)
    print('SYNTHETIC scheduler:',root,'22 core calls; 4 releases; cap path 16 calls/incomplete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path,nargs='?');p.add_argument('--source',type=Path)
    p.add_argument('--phase',choices=['factorial','released'],default='factorial')
    p.add_argument('--model-root',type=Path,default=Path('/mnt/models/hf-models'))
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:dry_run()
    else:
        assert a.destination and a.source
        h=None
        try:
            h=Harness(a.destination,a.model_root,'ple-'+a.phase)
            h.limit.cap=16 if a.phase=='released' else 24
            h.summary.update(protocol=__doc__,runner_sha256=sha(Path(__file__)),
                helper_sha256={n:sha(Path(__file__).with_name(n)) for n in ['run_nf4_asymmetry.py','run_nf4_local_bench.py']})
            session=read(a.destination/'session.json');session.update(protocol=__doc__,runner_sha256=h.summary['runner_sha256'],helper_sha256=h.summary['helper_sha256'])
            write(a.destination/'session.json',session);write(a.destination/'summary.json',h.summary)
            if a.phase=='factorial':factorial(h,a.source)
            else:released(h,a.source)
        except Exception as exc:
            if h is not None:h.finish(exc)
            elif (a.destination/'summary.json').exists():
                failed=read(a.destination/'summary.json');failed.update(status='failed',error=str(exc),traceback=traceback.format_exc())
                write(a.destination/'summary.json',failed)
            raise
        else:h.finish()
