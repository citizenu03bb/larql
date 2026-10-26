"""Frozen NF4 asymmetry transfer protocol, 2026-10-26, before model forwards.

Purpose: decide whether the Q4K symbol/number asymmetry has a local NF4 analogue
before interpreting a PLE intervention. One observed question per attribute;
no gain search, new donor, new layer or robustness/selectivity claim.

Eligibility service: <=16 forwards, freely generate hydrogen's atomic number
from empty chat answer to EOS. Require first value digit 1 and EOS. Symbol
ancestry comes from the verified NF4 bench's complete native H response.
Intervention service: <=24 forwards. At both saved native value boundaries:
native replay, full helium-subject donor (same native assistant prefix), zero,
R, random-R. Capture post-block L21 and the final projected PLE inputs at the
entity row of all 42 blocks. Reuse saved answer ids; this is a resumed-boundary
diagnostic, NOT an integrated edited generation from an empty answer.

R rule fixed before data: donor-minus-native post-L21 entity vector, multiplied
by float32(norm(delta[prompt entity..end]) / norm(delta[entity])). Derive this
once on the symbol question, never copy the Q4K number 2.585. Reuse EXACTLY this
vector on number and off-targets. Entity row must be one token at position 5.
Record whether symbol/number causal rows are bit-identical; require relative
L2 discrepancy <=0.001 for both native and donor (float16 shape-dependent
arithmetic). This tolerance is NOT used for same-input replays or zeros.
The reused intervention vector itself is always exactly identical. NF4 cast is
float16: random Gaussian seed 20261026 is matched to the ACTUAL float16 R norm;
allow <=0.1% cast mismatch, with no iterative correction or gain adjustment.

Two off-target questions, lithium/carbon symbol: native, R and random-R (6
forwards). Their value prefixes are explicitly supplied by substituting the
element name in the saved native symbol prefix. They are not generated answers
and test only local value-boundary leakage, not full-chat preservation.
Core total: 16 forwards. Zeros and native source replays must be bit-identical.

Transfer gate, fixed before data: full donors select He/2; R selects He for
symbol and 1 for number; symbol margin increment >0, >2*max(random increment,0)
and >2*max(number increment,0). If this gate fails after core, STOP transfer;
no PLE test or gain search. If it passes, release both R branches for <=4
tokens each (up to 24 total). Require EOS, no helium in the symbol continuation,
and number still 1. A cap is inconclusive and does not authorize a PLE test.
No causal interpretation of off-target preservation from two supplied prefixes.

Total A budget <=40 forwards across TWO sequential services, <=24 per service,
each 22G/24G/8G, 1800 s (runner deadline 1740 s). One E4B worker at a time.
Stop on OOM, nonfinite data, budget/deadline, tokenization/ancestry/instrument
failure; report incomplete. No retries with different experimental parameters.
Full saved arrays, hashes, model-free same-author checker, report and archive.
PLE test is conditional on a verified positive transfer, with a separate frozen
protocol before its first forward. No weight/vindex/persistent patch writes.
"""
import argparse
import contextlib
import json
import importlib.metadata
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from run_nf4_local_bench import Limit, array, setup, sha, write, SEED, QUESTION

NUMBER = "Give hydrogen's atomic number in one short sentence, with the element's name before the number."


@contextlib.contextmanager
def hooks(layers, position=5, residual=None, ple_replacement=None, ple_addition=None):
    data, handles, seen = {}, [], set()
    ple_before, ple_after = {}, {}
    for index, layer in enumerate(layers):
        def pre(module, args, kwargs, index=index):
            assert index not in seen, 'duplicate decoder pre-hook'
            seen.add(index)
            assert len(args) >= 2 and isinstance(args[1], torch.Tensor)
            value = args[1]
            ple_before[index] = array(value[0, position])
            if ple_replacement is not None or ple_addition is not None:
                patched = value.clone()
                if ple_replacement is not None:
                    patched[0, position] = torch.as_tensor(ple_replacement[index], device=value.device, dtype=value.dtype)
                else:
                    patched[0, position] += torch.as_tensor(ple_addition[index], device=value.device, dtype=value.dtype)
                assert torch.equal(value[:, :position], patched[:, :position])
                assert torch.equal(value[:, position + 1:], patched[:, position + 1:])
                ple_after[index] = array(patched[0, position])
                args = (args[0], patched, *args[2:])
            return args, kwargs
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    def post(module, args, output):
        assert 'l21_before' not in data
        data['l21_before'] = array(output[0])
        if residual is not None:
            patched = output.clone()
            delta = torch.as_tensor(residual, device=output.device, dtype=output.dtype)
            patched[0, position] += delta
            assert torch.equal(output[:, :position], patched[:, :position])
            assert torch.equal(output[:, position + 1:], patched[:, position + 1:])
            data['requested_residual'] = array(delta)
            data['l21_after'] = array(patched[0])
            return patched
    handles.append(layers[21].register_forward_hook(post))
    try:
        yield data
        assert seen == set(range(len(layers)))
        data['ple_before'] = np.stack([ple_before[i] for i in range(len(layers))])
        if ple_after:
            data['ple_after'] = np.stack([ple_after[i] for i in range(len(layers))])
    finally:
        for handle in handles:
            handle.remove()


class Harness:
    def __init__(self, root, model_root, phase):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self.limit = Limit(cap=16 if phase == 'eligibility' else 24)
        self.summary = dict(status='running', phase=phase, synthetic=False, calls=[], forwards=0,
                            protocol=__doc__, runner_sha256=sha(Path(__file__)))
        write(root/'summary.json', self.summary)
        torch.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.model, self.layers, self.prompt, self.eos, self.decode, session, _ = setup('E4B', model_root, False)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_root/'gemma-4-E4B-it', local_files_only=True)
        session.update(protocol=__doc__, runner_sha256=self.summary['runner_sha256'],
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,text=True).strip(),
            eos_ids=sorted(self.eos), versions={n:importlib.metadata.version(n) for n in [
                'torch','transformers','bitsandbytes','accelerate','numpy']})
        write(root/'session.json', session)

    def prompt_for(self, question):
        return self.tok.apply_chat_template([{'role':'user','content':question}],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)

    def token(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        assert len(ids) == 1, (text, ids)
        return ids[0]

    def forward(self, name, ids, residual=None, ple_replacement=None, ple_addition=None):
        self.limit.take()
        ordinal = self.limit.count-1
        tensor = torch.tensor([ids], device='cuda', dtype=torch.long)
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.inference_mode(), hooks(self.layers, residual=residual,
                ple_replacement=ple_replacement, ple_addition=ple_addition) as data:
            out = self.model(input_ids=tensor, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            seconds = time.monotonic()-start
            data['logits'] = array(out.logits[0,-1])
        assert all(np.isfinite(x).all() for x in data.values())
        filename = f'{ordinal:03}-{name}.npz'
        np.savez_compressed(self.root/filename, **data)
        rec = dict(name=name, ordinal=ordinal, input_ids=list(ids), seconds=seconds,
            artifact=filename, artifact_sha256=sha(self.root/filename),
            argmax=int(data['logits'].argmax()), residual=residual is not None,
            ple_replacement=ple_replacement is not None, ple_addition=ple_addition is not None)
        self.summary['calls'].append(rec);self.summary['forwards']=self.limit.count
        write(self.root/'summary.json',self.summary)
        print(json.dumps(rec),flush=True)
        return rec,data

    def finish(self, error=None):
        if error is not None:
            self.summary.update(status='failed',error=str(error),traceback=traceback.format_exc())
        cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
        self.summary['cgroup']={n:(cg/n).read_text().strip() for n in [
            'memory.peak','memory.swap.peak','memory.high','memory.max','memory.swap.max'] if (cg/n).exists()}
        self.summary['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        write(self.root/'summary.json',self.summary)


def saved_call(root, name):
    summary=json.loads((root/'summary.json').read_text())
    rec=next(c for c in summary['calls'] if c['name']==name)
    assert sha(root/rec['artifact'])==rec['artifact_sha256']
    with np.load(root/rec['artifact'],allow_pickle=False) as z:
        return rec,{k:z[k].copy() for k in z.files}


def source_symbol(bench):
    s=json.loads((bench/'summary.json').read_text());v=json.loads((bench/'verification.json').read_text())
    assert v['status']=='verified_complete' and v['native_h_observed'] and s['native']['eos']
    candidates=[name for name,token in zip(s['native']['call_names'],s['native']['ids']) if token==236814]
    assert len(candidates)==1
    rec, data=saved_call(bench,candidates[0])
    prompt=json.loads((bench/'session.json').read_text())['prompt_ids']
    return prompt,rec,data


def margin(data, target, native):
    return float(data['logits'][target])-float(data['logits'][native])


def eligibility(h):
    prompt=h.prompt_for(NUMBER);seed=[];names=[];boundary=None
    assert prompt[:6]==h.prompt[:6] and h.tok.decode([prompt[5]])==' hydrogen'
    for i in range(16):
        rec,data=h.forward(f'number_native{i:02}',prompt+seed)
        piece=h.tok.decode([rec['argmax']]).strip()
        if boundary is None and piece in set('0123456789') and len(piece)==1:
            boundary=rec['name']
        seed.append(rec['argmax']);names.append(rec['name'])
        if seed[-1] in h.eos:break
    complete=seed[-1] in h.eos
    value=next((c['argmax'] for c in h.summary['calls'] if c['name']==boundary),None)
    good=bool(complete and value==h.token('1') and "hydrogen" in h.decode(seed).lower())
    h.summary.update(status='complete' if complete else 'incomplete_native_cap',
        native=dict(prompt_ids=prompt,ids=seed,text=h.decode(seed),call_names=names,eos=complete,boundary=boundary),
        eligible=good)


def interventions(h, bench, native_root):
    prompt, symbol_src, symbol_saved=source_symbol(bench)
    assert prompt==h.prompt
    n=json.loads((native_root/'summary.json').read_text())
    nv=json.loads((native_root/'verification.json').read_text())
    assert n['eligible'] and nv['status']=='verified_complete'
    number_src,number_saved=saved_call(native_root,n['native']['boundary'])
    n_prompt=n['native']['prompt_ids']
    questions=[dict(name='symbol',prompt=prompt,source=symbol_src,saved=symbol_saved,
                    native=h.token('H'),target=h.token('He')),
               dict(name='number',prompt=n_prompt,source=number_src,saved=number_saved,
                    native=h.token('1'),target=h.token('2'))]
    cases={};delta=None;random=None
    for q in questions:
        name=q['name'];ids=q['source']['input_ids'];donor=list(ids);donor[5]=h.token(' helium')
        base_rec,base=h.forward(name+'_base',ids)
        assert np.array_equal(base['logits'],q['saved']['logits'].reshape(-1)), 'native source replay differs'
        assert base_rec['argmax']==q['native']
        _,full=h.forward(name+'_donor',donor)
        if delta is None:
            contrast=full['l21_before']-base['l21_before']
            entity=contrast[5];prompt_delta=contrast[5:len(prompt)]
            gain=np.float32(np.linalg.norm(prompt_delta.astype(np.float64))/np.linalg.norm(entity.astype(np.float64)))
            delta=(gain*entity).astype(np.float32)
            actual_norm=float(np.linalg.norm(delta.astype(np.float16).astype(np.float64)))
            draw=np.random.default_rng(SEED).standard_normal(delta.shape)
            random=(draw/np.linalg.norm(draw)*actual_norm).astype(np.float32)
            assert abs(np.linalg.norm(random.astype(np.float16).astype(np.float64))/actual_norm-1)<=0.001
            np.save(h.root/'residual.npy',delta);np.save(h.root/'random.npy',random)
            h.summary['norm_rule']=dict(gain=float(gain),actual_norm=actual_norm,seed=SEED,prompt_end=len(prompt))
        else:
            row_check={}
            for label,value in [('base',base),('donor',full)]:
                reference=cases['symbol'][label]['l21_before'][5].astype(np.float64)
                observed=value['l21_before'][5].astype(np.float64)
                relative=float(np.linalg.norm(observed-reference)/np.linalg.norm(reference))
                assert relative<=0.001, 'causal-row discrepancy exceeds frozen float16 tolerance'
                row_check[label]=dict(bit_identical=np.array_equal(observed,reference),relative_l2=relative)
            h.summary['causal_row_check']=row_check
        _,zero=h.forward(name+'_zero',ids,residual=np.zeros_like(delta))
        assert np.array_equal(zero['logits'],base['logits'])
        r_rec,r=h.forward(name+'_residual',ids,residual=delta)
        _,rand=h.forward(name+'_random',ids,residual=random)
        cases[name]=dict(base=base,donor=full,residual=r,random=rand,record=r_rec,question=q)
    seed=symbol_src['input_ids'][len(prompt):]
    off=[]
    for name in ('lithium','carbon'):
        p=h.prompt_for(QUESTION.replace('hydrogen',name))
        assert len(p)==len(prompt) and p[:5]==prompt[:5] and p[6:]==prompt[6:]
        supplied=h.tok.encode(h.tok.decode(seed).replace('Hydrogen',name.capitalize(),1),add_special_tokens=False)
        ids=p+supplied
        base_rec,base=h.forward(name+'_base',ids)
        r_rec,r=h.forward(name+'_residual',ids,residual=delta)
        random_rec,rand=h.forward(name+'_random',ids,residual=random)
        off.append(dict(name=name,prompt_ids=p,supplied_seed_ids=supplied,
                        baseline_argmax=base_rec['argmax'],residual_argmax=r_rec['argmax'],random_argmax=random_rec['argmax']))
    metrics={}
    for name,c in cases.items():
        q=c['question'];bm=margin(c['base'],q['target'],q['native'])
        metrics[name]=dict(native_id=q['native'],target_id=q['target'],baseline_margin=bm,
            residual_margin=margin(c['residual'],q['target'],q['native']),
            residual_increment=margin(c['residual'],q['target'],q['native'])-bm,
            random_increment=margin(c['random'],q['target'],q['native'])-bm,
            residual_argmax=c['record']['argmax'],donor_argmax=int(c['donor']['logits'].argmax()))
    s,num=metrics['symbol'],metrics['number']
    gate=bool(s['donor_argmax']==s['target_id'] and num['donor_argmax']==num['target_id']
        and s['residual_argmax']==s['target_id'] and num['residual_argmax']==num['native_id']
        and s['residual_increment']>0 and s['residual_increment']>2*max(s['random_increment'],0)
        and s['residual_increment']>2*max(num['residual_increment'],0))
    h.summary.update(metrics=metrics,off_target=off,core_gate=gate,
        source_symbol_summary_sha256=sha(bench/'summary.json'),source_number_summary_sha256=sha(native_root/'summary.json'))
    continuations={}
    if gate:
        for name,c in cases.items():
            q=c['question'];prefix=q['source']['input_ids'];emitted=[c['record']['argmax']];calls=[c['record']['name']]
            for i in range(4):
                if emitted[-1] in h.eos:break
                rec,_=h.forward(f'{name}_released{i}',prefix+emitted)
                emitted.append(rec['argmax']);calls.append(rec['name'])
            seed=prefix[len(q['prompt']):]+emitted
            continuations[name]=dict(ids=emitted,call_names=calls,text=h.decode(seed),eos=emitted[-1] in h.eos)
    ready=bool(gate and all(c['eos'] for c in continuations.values())
               and 'helium' not in continuations['symbol']['text'].lower())
    h.summary.update(status='complete',continuations=continuations,transfer_reproduced=ready,
                     interpretation='local NF4 asymmetry reproduced' if ready else 'STOP: no qualified NF4 transfer; no gain search or PLE run')


def dry_run():
    class Layer(torch.nn.Module):
        def forward(self,h,p):return h+p.repeat(1,1,h.shape[-1]//p.shape[-1])*0.01
    layers=torch.nn.ModuleList(Layer() for _ in range(42))
    initial=torch.arange(8*8,dtype=torch.float32).reshape(1,8,8)/64
    ple=torch.arange(42*8*2,dtype=torch.float32).reshape(42,1,8,2)/100
    def one(r=None,p=None,a=None):
        with hooks(layers,residual=r,ple_replacement=p,ple_addition=a) as data:
            h=initial.clone()
            for i,l in enumerate(layers):h=l(h,ple[i])
        return h,data
    base,b=one();zero,z=one(np.zeros(8,dtype=np.float32));assert torch.equal(base,zero)
    delta=np.ones(8,dtype=np.float32)*0.125
    edited,e=one(delta);assert torch.equal(base[:,:5],edited[:,:5]) and torch.equal(base[:,6:],edited[:,6:])
    replacement=b['ple_before']+0.25
    changed,p=one(p=replacement);assert np.array_equal(p['ple_after'],replacement)
    assert torch.equal(base[:,:5],changed[:,:5]) and torch.equal(base[:,6:],changed[:,6:])
    plus,pa=one(a=np.zeros_like(replacement));assert torch.equal(base,plus)
    assert np.array_equal(e['l21_after'][5],e['l21_before'][5]+delta)
    limit=Limit(cap=2);limit.take();limit.take()
    try:limit.take()
    except RuntimeError:pass
    else:raise AssertionError('cap bypass')
    print('SYNTHETIC dry-run: row support, exact residual/PLE application, zeros, cleanup and hard cap pass')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path,nargs='?')
    p.add_argument('--phase',choices=['eligibility','interventions'])
    p.add_argument('--model-root',type=Path,default=Path('/mnt/models/hf-models'))
    p.add_argument('--bench-source',type=Path)
    p.add_argument('--native-source',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:dry_run()
    else:
        assert a.destination and a.phase
        h=None
        try:
            h=Harness(a.destination,a.model_root,a.phase)
            if a.phase=='eligibility':eligibility(h)
            else:interventions(h,a.bench_source,a.native_source)
        except Exception as e:
            if h is not None:h.finish(e)
            elif (a.destination/'summary.json').exists():
                failed=json.loads((a.destination/'summary.json').read_text())
                failed.update(status='failed',error=str(e),traceback=traceback.format_exc())
                write(a.destination/'summary.json',failed)
            raise
        else:h.finish()
