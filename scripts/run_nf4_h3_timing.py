"""H3 timing protocol, frozen 2026-10-26 before any new model forward.

Mandate: test whether an earlier entity-row intervention influences atomic
number more than the already tested L21 intervention. It serves an honest
mechanistic showcase, not persistent editing. H1's registered result remains
undetermined; its 8x8 donor-symbol matrix is post hoc. R and PLE influencing
symbol is not proof of identity transport or redundant pathways.

Fixed sites: POST-BLOCK L8, L14, L21 (zero based), row 5 only, one pulse per
forward, no cumulative transplant. Same stored native symbol and number value
boundaries and H/He, one/two readouts from nf4-asymmetry-20261026. No prompt or
answer-prefix retyping. Complete helium subject differs only at prompt row 5.
PLE inputs remain native; capture them to verify this.

Norm rule: common requested float16 L2 norm N=norm(archived residual.npy cast
float16), about 63.8967. Native/donor forwards capture entity rows at all three
sites. At L8/L14 take the symbol donor-minus-native vector, normalize ONCE to
N in float64, cast float32 then float16 at application. At L21 reuse archived
residual.npy EXACTLY. Same vectors on both attributes and off-targets. Reuse
archived random.npy, one fixed Gaussian direction, at each site/attribute.
Requested cast norms must match N within 0.1%; actual rounded row-change norms
must match within 1%. No gain search. Record native-row norms and relative
doses: equal L2 energy does not imply equal layer sensitivity.

Core, EXACTLY 24 forwards: for symbol and number native+full donor (4);
R and matched random at each of 3 sites on both attributes (12); zero at all
3 sites on symbol and at L21 on number (4); lithium/carbon supplied-prefix
symbol native+L8-R (4). Shared symbol zeros validate each hook; number has its
own native replay and L21 zero. Each input-shape intervention addition is also
checked exactly. Off-targets cover local leakage only, not full-chat selectivity.
Native, full donor and L21 R/random must replay archived full logits bitwise.
Causal entity rows across attributes must agree to <=0.001 relative L2;
record exact equality separately, never use tolerance for same-input replays.

Read E_attr,L = margin(R_L)-margin(native), X_attr,L likewise random. Margins
are target-minus-native logits, top-1 secondary. Early site k has a specific
number improvement iff E_N,k>=1, E_N,k-E_N,21>=1, E_N,k>2*max(X_N,k,0).
It matches the relative-timing pattern iff also E_S,k<=E_S,21+1 (symbol does
not grow materially). Frozen readings:
- H3-compatible ordered pattern: BOTH early sites match, E_N,8>=E_N,14+1 and
  E_N,14>=E_N,21+1.
- partial/local early-site compatibility: at least one site matches, but the
  complete ordered pattern fails; never call this monotone timing established.
- earlier number sensitivity without relative selectivity: a number improvement
  passes but neither site matches the symbol condition.
- otherwise no support at these sites/dose. No follow-up layer/gain search.
Report both sites, the early-minus-L21 changes for both attributes, and their
difference; no p-values or unique K/V-mediation claim. Earlier pulses alter
subsequent residual history AND K/V, so they cannot isolate the K/V route.

Released phase: inspect every changed R top-1 among the SIX primary cases and
TWO L8 off-target cases. Fixed order number L8/L14/L21, symbol L8/L14/L21,
lithium L8, carbon L8. Reuse the actual selected token and remove ALL patches;
<=3 more tokens or EOS per case, <=24 forwards in a separate service. A cap
stays incomplete; never extend it. These are resumed-prefix chains, not
integrated edited-from-empty demonstrations. Random controls remain logit-only.

Budget <=48 new forwards, <=24 per service, each 22G/24G/8G, 1800 s (internal
deadline 1740 s), ONE heavy worker. Before EVERY NF4/LARQL service acquire the
SHARED .heavy-model.lock via flock -w 3600 through systemd-run --user --wait.
Synthetic full-scheduler/hook/cap test and this committed protocol precede
launch. Stop on cap/deadline/OOM/nonfinite/replay/norm/ancestry failure. No
persistent model/vindex/patch changes; no push or publication in this mandate.
"""
import argparse
import contextlib
import json
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from run_nf4_asymmetry import Harness, saved_call
from run_nf4_local_bench import array, sha, write

SITES=(8,14,21)
RELEASE_ORDER=tuple(f'{q}_L{k:02}' for q in ('number','symbol') for k in SITES)+('lithium_L08','carbon_L08')


@contextlib.contextmanager
def timing_hooks(layers,site=None,vector=None):
    data,handles,seen,ple={ },[],set(),{}
    for i,layer in enumerate(layers):
        def pre(module,args,kwargs,i=i):
            assert i not in seen and len(args)>=2
            seen.add(i);ple[i]=array(args[1][0,5])
        handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
    for k in SITES:
        def post(module,args,out,k=k):
            assert f'entity_L{k:02}' not in data
            data[f'entity_L{k:02}']=array(out[0,5])
            if k==site:
                patched=out.clone();v=torch.as_tensor(vector,dtype=out.dtype,device=out.device)
                patched[0,5]+=v
                assert torch.equal(out[:,:5],patched[:,:5]) and torch.equal(out[:,6:],patched[:,6:])
                data['patch_before']=array(out[0]);data['patch_after']=array(patched[0]);data['requested_delta']=array(v)
                return patched
        handles.append(layers[k].register_forward_hook(post))
    try:
        yield data
        assert seen==set(range(len(layers)))
        data['ple_before']=np.stack([ple[i] for i in range(len(layers))])
    finally:
        for h in handles:h.remove()


class TimingHarness(Harness):
    def forward(self,name,ids,site=None,vector=None):
        self.limit.take();ordinal=self.limit.count-1
        tensor=torch.tensor([ids],dtype=torch.long,device='cuda')
        torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode(),timing_hooks(self.layers,site,vector) as data:
            out=self.model(input_ids=tensor,use_cache=False,logits_to_keep=1)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            data['logits']=array(out.logits[0,-1])
        assert all(np.isfinite(x).all() for x in data.values())
        fn=f'{ordinal:03}-{name}.npz';np.savez_compressed(self.root/fn,**data)
        rec=dict(ordinal=ordinal,name=name,input_ids=list(ids),artifact=fn,artifact_sha256=sha(self.root/fn),
                 argmax=int(data['logits'].argmax()),patch_layer=site,seconds=seconds)
        self.summary['calls'].append(rec);self.summary['forwards']=self.limit.count
        write(self.root/'summary.json',self.summary);print(json.dumps(rec),flush=True)
        return rec,data


def read(p):return json.loads(p.read_text())
def margin(x,t,n):return float(x['logits'][t])-float(x['logits'][n])


def readings(metrics):
    n=metrics['number'];s=metrics['symbol'];sites={}
    for k in (8,14):
        key=str(k);en=n[key]['effect'];rn=n[key]['random_effect']
        improvement=en-n['21']['effect'];se=s[key]['effect']-s['21']['effect']
        specific=bool(en>=1 and improvement>=1 and en>2*max(rn,0))
        sites[key]=dict(number_specific_improvement=specific,number_early_minus_L21=improvement,
            symbol_early_minus_L21=se,relative_contrast=improvement-se,
            timing_pattern=bool(specific and se<=1))
    ordered=bool(sites['8']['timing_pattern'] and sites['14']['timing_pattern']
                 and n['8']['effect']>=n['14']['effect']+1 and n['14']['effect']>=n['21']['effect']+1)
    if ordered:label='H3-compatible ordered pattern at these sites and common dose'
    elif any(x['timing_pattern'] for x in sites.values()):label='Partial/local early-site compatibility; complete ordered pattern not met'
    elif any(x['number_specific_improvement'] for x in sites.values()):label='Earlier number sensitivity without the relative symbol pattern'
    else:label='No H3 support at these sites and common dose; no general timing falsification'
    return dict(sites=sites,ordered_pattern=ordered,reading=label)


def norm_stats(before,after,requested,N):
    cast=float(np.linalg.norm(requested.astype(np.float64)))
    actual=float(np.linalg.norm(after[5].astype(np.float64)-before[5].astype(np.float64)))
    assert abs(cast/N-1)<=0.001 and abs(actual/N-1)<=0.01
    native=float(np.linalg.norm(before[5].astype(np.float64)))
    return dict(cast_norm=cast,actual_norm=actual,native_row_norm=native,relative_dose=actual/native)


def core(h,source):
    prior=read(source/'summary.json');receipt=read(source/'verification.json')
    assert prior['synthetic']==h.summary['synthetic'] and receipt['status']=='verified_complete'
    oldR=np.load(source/'residual.npy',allow_pickle=False);random=np.load(source/'random.npy',allow_pickle=False)
    N=float(np.linalg.norm(oldR.astype(np.float16).astype(np.float64)))
    assert abs(np.linalg.norm(random.astype(np.float16).astype(np.float64))/N-1)<=0.001
    h.summary.update(source_summary_sha256=sha(source/'summary.json'),source_verification_sha256=sha(source/'verification.json'),
        common_norm=N,sites=list(SITES),metrics={},norms={},release_cases=[],off_target=[])
    np.save(h.root/'random.npy',random);vectors={};calibration={}
    for q in ('symbol','number'):
        old,b0=saved_call(source,q+'_base');donor,d0=saved_call(source,q+'_donor')
        br,b=h.forward(q+'_base',old['input_ids']);dr,d=h.forward(q+'_donor',donor['input_ids'])
        assert np.array_equal(b['logits'],b0['logits']) and np.array_equal(d['logits'],d0['logits'])
        assert np.array_equal(b['ple_before'],b0['ple_before']) and np.array_equal(d['ple_before'],d0['ple_before'])
        if q=='symbol':
            calibration={'base':b,'donor':d}
            for k in SITES:
                e=d[f'entity_L{k:02}']-b[f'entity_L{k:02}']
                vectors[k]=oldR.copy() if k==21 else (e.astype(np.float64)/np.linalg.norm(e.astype(np.float64))*N).astype(np.float32)
                assert abs(np.linalg.norm(vectors[k].astype(np.float16).astype(np.float64))/N-1)<=0.001
                np.save(h.root/f'residual-L{k:02}.npy',vectors[k])
        else:
            causal={}
            for k in SITES:
                causal[str(k)]={}
                for label,x in [('base',b),('donor',d)]:
                    ref=calibration[label][f'entity_L{k:02}'].astype(np.float64);value=x[f'entity_L{k:02}'].astype(np.float64)
                    rel=float(np.linalg.norm(value-ref)/np.linalg.norm(ref));assert rel<=0.001
                    causal[str(k)][label]=dict(bit_identical=np.array_equal(ref,value),relative_l2=rel)
            h.summary['causal_rows']=causal
        for k in (SITES if q=='symbol' else (21,)):
            _,zero=h.forward(q+f'_zero_L{k:02}',old['input_ids'],site=k,vector=np.zeros_like(oldR))
            assert np.array_equal(zero['logits'],b['logits'])
        native=prior['metrics'][q]['native_id'];target=prior['metrics'][q]['target_id'];bm=margin(b,target,native)
        h.summary['metrics'][q]={}
        for k in SITES:
            rec,r=h.forward(q+f'_L{k:02}',old['input_ids'],site=k,vector=vectors[k])
            _,x=h.forward(q+f'_random_L{k:02}',old['input_ids'],site=k,vector=random)
            assert np.array_equal(r['ple_before'],b['ple_before']) and np.array_equal(x['ple_before'],b['ple_before'])
            h.summary['norms'][q+f'_L{k:02}']=norm_stats(r['patch_before'],r['patch_after'],r['requested_delta'],N)
            h.summary['norms'][q+f'_random_L{k:02}']=norm_stats(x['patch_before'],x['patch_after'],x['requested_delta'],N)
            if k==21:
                _,oldr=saved_call(source,q+'_residual');_,oldx=saved_call(source,q+'_random')
                assert np.array_equal(r['logits'],oldr['logits']) and np.array_equal(x['logits'],oldx['logits'])
            h.summary['metrics'][q][str(k)]=dict(native_id=native,target_id=target,baseline_margin=bm,
                margin=margin(r,target,native),effect=margin(r,target,native)-bm,random_effect=margin(x,target,native)-bm,
                argmax=rec['argmax'],random_argmax=int(x['logits'].argmax()),donor_argmax=dr['argmax'])
            if rec['argmax']!=native:h.summary['release_cases'].append(dict(name=q+f'_L{k:02}',question=q,
                prefix_ids=rec['input_ids'],first_token=rec['argmax'],native_id=native,target_id=target,
                source_artifact=rec['artifact'],source_artifact_sha256=rec['artifact_sha256']))
    for off in prior['off_target']:
        q=off['name'];old,b0=saved_call(source,q+'_base')
        br,b=h.forward(q+'_base',old['input_ids']);assert np.array_equal(b['logits'],b0['logits'])
        rr,r=h.forward(q+'_L08',old['input_ids'],site=8,vector=vectors[8])
        assert np.array_equal(r['ple_before'],b['ple_before'])
        h.summary['norms'][q+'_L08']=norm_stats(r['patch_before'],r['patch_after'],r['requested_delta'],N)
        h.summary['off_target'].append(dict(name=q,prefix_type='supplied',native_argmax=br['argmax'],residual_argmax=rr['argmax']))
        if rr['argmax']!=br['argmax']:h.summary['release_cases'].append(dict(name=q+'_L08',question=q,
            prefix_ids=rr['input_ids'],first_token=rr['argmax'],native_id=br['argmax'],target_id=None,
            source_artifact=rr['artifact'],source_artifact_sha256=rr['artifact_sha256']))
    h.summary['release_cases'].sort(key=lambda x:RELEASE_ORDER.index(x['name']))
    h.summary.update(status='complete',readings=readings(h.summary['metrics']))


def released(h,source):
    s=read(source/'summary.json');v=read(source/'verification.json')
    assert s['synthetic']==h.summary['synthetic'] and v['status']=='verified_complete' and s['release_cases']
    h.summary.update(source_summary_sha256=sha(source/'summary.json'),branches=[])
    for case in s['release_cases']:
        assert sha(source/case['source_artifact'])==case['source_artifact_sha256']
        emitted=[case['first_token']];names=[]
        for i in range(3):
            if emitted[-1] in h.eos:break
            rec,_=h.forward(case['name']+f'_released{i}',case['prefix_ids']+emitted)
            emitted.append(rec['argmax']);names.append(rec['name'])
        ids=case['prefix_ids'];marker=[105,4368,107];starts=[i+3 for i in range(len(ids)-2) if ids[i:i+3]==marker];assert len(starts)==1
        h.summary['branches'].append(dict(name=case['name'],ids=emitted,call_names=names,
            text=h.decode(ids[starts[0]:]+emitted),eos=emitted[-1] in h.eos))
    h.summary['status']='complete' if all(b['eos'] for b in h.summary['branches']) else 'incomplete_released_cap'


def dry_run():
    class Layer(torch.nn.Module):
        def forward(self,h,p):return h+p*0.01
    layers=torch.nn.ModuleList(Layer() for _ in range(42));initial=torch.arange(80,dtype=torch.float32).reshape(1,10,8)/100
    ple=torch.zeros((1,10,8));delta=np.ones(8,np.float32)*0.125
    def one(k=None,v=None):
        with timing_hooks(layers,k,v) as d:
            h=initial.clone()
            for l in layers:h=l(h,ple)
        return h,d
    base,_=one()
    for k in SITES:
        zero,_=one(k,np.zeros(8,np.float32));assert torch.equal(zero,base)
        patched,d=one(k,delta);assert torch.equal(patched[:,:5],base[:,:5]) and torch.equal(patched[:,6:],base[:,6:])
        assert np.array_equal(d['patch_after'][5],d['patch_before'][5]+delta)
    synthetic_scheduler()
    print('SYNTHETIC H3 hooks, shared zeros, scheduler and release caps passed; no pretrained-model evidence')


def synthetic_scheduler():
    root=Path(tempfile.mkdtemp(prefix='larql-h3-synthetic-',dir='/tmp'));source=root/'source';source.mkdir()
    oldR=np.ones(8,np.float32)*0.5;random=np.ones(8,np.float32)*-0.5;np.save(source/'residual.npy',oldR);np.save(source/'random.npy',random)
    def synth(q,site=None,v=None,donor=False,end=False):
        native,target=(0,1) if q==0 else (2,3) if q==2 else (4,1) if q==4 else (5,1)
        logits=np.zeros(16,np.float32);logits[native]=1;logits[target]=0.99
        if donor:logits[target]=2
        elif site is not None and np.any(v):
            if np.mean(v)<0:effect=0.001
            else:effect=({8:10,14:5,21:0.05} if q==2 else {8:9,14:12,21:19.7})[site]
            logits[target]+=effect
        if end:logits[:]=0;logits[9]=3
        b=np.full((13,8),1 if donor else 0.5,np.float32);d={'logits':logits,'ple_before':np.zeros((42,8),np.float32)}
        for k in SITES:d[f'entity_L{k:02}']=np.ones(8,np.float32)*(1 if donor else 0.5)
        if site is not None:d.update(patch_before=b,patch_after=b.copy(),requested_delta=v.copy());d['patch_after'][5]=b[5]+v
        return d
    class Fake:
        def __init__(self,p,never=False):
            self.root=p;p.mkdir();self.eos={9};self.never=never;self.limit=SimpleNamespace(count=0)
            self.summary=dict(status='running',synthetic=True,calls=[],forwards=0)
        def forward(self,name,ids,site=None,vector=None):
            x=synth(ids[4],site,vector,ids[5]==99,len(ids)>13 and not self.never)
            f=f'{self.limit.count:03}-{name}.npz';np.savez_compressed(self.root/f,**x)
            rec=dict(name=name,input_ids=ids.copy(),artifact=f,artifact_sha256=sha(self.root/f),argmax=int(x['logits'].argmax()))
            self.summary['calls'].append(rec);self.limit.count+=1;self.summary['forwards']=self.limit.count
            return rec,x
        def decode(self,ids):return ' '.join(map(str,ids))
    calls=[]
    for name,q in [('symbol',0),('number',2),('lithium',4),('carbon',5)]:
        for mode in (('base','donor','residual','random') if q<4 else ('base',)):
            ids=[2,105,2364,107,q,99 if mode=='donor' else 10,1,106,107,105,4368,107,42]
            x=synth(q,21 if mode in ('residual','random') else None,oldR if mode=='residual' else random if mode=='random' else None,mode=='donor')
            fn=name+'_'+mode+'.npz';np.savez_compressed(source/fn,**x)
            calls.append(dict(name=name+'_'+mode,input_ids=ids,artifact=fn,artifact_sha256=sha(source/fn)))
    write(source/'summary.json',dict(status='complete',synthetic=True,calls=calls,metrics={'symbol':{'native_id':0,'target_id':1},'number':{'native_id':2,'target_id':3}},off_target=[{'name':'lithium'},{'name':'carbon'}]))
    write(source/'verification.json',{'status':'verified_complete','synthetic':True})
    h=Fake(root/'core');core(h,source);assert h.limit.count==24 and len(h.summary['release_cases'])==8
    assert h.summary['readings']['ordered_pattern']
    write(h.root/'summary.json',h.summary);write(h.root/'verification.json',{'status':'verified_complete','synthetic':True})
    h2=Fake(root/'released');released(h2,h.root);assert h2.limit.count==8 and h2.summary['status']=='complete'
    cap=Fake(root/'released-cap',True);released(cap,h.root);assert cap.limit.count==24 and cap.summary['status']=='incomplete_released_cap'
    for f in (h2,cap):write(f.root/'summary.json',f.summary)
    print('SYNTHETIC scheduler:',root,'24 core calls; 8 changed cases; 24-call incomplete cap path')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path,nargs='?')
    p.add_argument('--source',type=Path);p.add_argument('--phase',choices=['core','released'],default='core')
    p.add_argument('--model-root',type=Path,default=Path('/mnt/models/hf-models'));p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:dry_run()
    else:
        assert a.destination and a.source;h=None
        try:
            h=TimingHarness(a.destination,a.model_root,'h3-'+a.phase)
            h.summary.update(protocol=__doc__,runner_sha256=sha(Path(__file__)),helper_sha256={n:sha(Path(__file__).with_name(n)) for n in ('run_nf4_asymmetry.py','run_nf4_local_bench.py')})
            session=read(a.destination/'session.json');session.update(protocol=__doc__,runner_sha256=h.summary['runner_sha256'],helper_sha256=h.summary['helper_sha256'])
            write(a.destination/'session.json',session);write(a.destination/'summary.json',h.summary)
            if a.phase=='core':core(h,a.source)
            else:released(h,a.source)
        except Exception as exc:
            if h is not None:h.finish(exc)
            elif (a.destination/'summary.json').exists():
                x=read(a.destination/'summary.json');x.update(status='failed',error=str(exc),traceback=traceback.format_exc());write(a.destination/'summary.json',x)
            raise
        else:h.finish()
