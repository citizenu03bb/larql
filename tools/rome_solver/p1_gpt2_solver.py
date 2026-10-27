"""Independent Codex GPT-2 XL ROME port, not an import of Claude's solve.

Published GPT-2 XL settings; P1 A1 fixes 20 actual Adam updates, without
loss-based early stopping. The key direction averages the shared contexts;
the affine constraint uses the clean context, as in published compute_v.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Example:
    text: str
    ids: list
    position: int
    target_ids: list


def example(tokenizer,text,subject,target=None):
    if text.count(subject)!=1:raise ValueError('Exactly one subject mention required')
    start=text.index(subject);end=start+len(subject)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    positions=[i for i,(a,b)in enumerate(encoded['offset_mapping'])if b>start and a<end]
    if not positions:raise ValueError('Subject has no token')
    ids=list(encoded['input_ids']);target_ids=[]
    if target is not None:
        target_ids=list(tokenizer.encode(target,add_special_tokens=False))
        full=list(tokenizer.encode(text+target,add_special_tokens=False))
        if not target_ids or full!=ids+target_ids:raise ValueError('Target changes prefix tokenization')
        ids=full[:-1]
    if not ids or len(ids)>128:raise ValueError('Rewrite outside frozen 128-token safety limit')
    return Example(text,ids,positions[-1],target_ids)


def batch(examples,pad,device):
    length=max(len(e.ids)for e in examples)
    ids=torch.full((len(examples),length),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,e in enumerate(examples):ids[i,:len(e.ids)]=torch.tensor(e.ids,device=device);mask[i,:len(e.ids)]=1
    return ids,mask


def hidden(model,examples,pad,counts):
    ids,mask=batch(examples,pad,next(model.parameters()).device);counts.take(len(examples))
    h=model.transformer(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
    if not torch.isfinite(h).all():raise ValueError('Nonfinite transformer output')
    return h


@contextmanager
def delta_at(mlp,positions,delta):
    def hook(mod,args,out):
        edited=out.clone();rows=torch.arange(len(positions),device=out.device)
        edited[rows,positions]=out[rows,positions]+delta.to(out.dtype)
        return edited
    handle=mlp.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def affine_output_factor(target,base,clean_key,input_factor):
    denominator=float(np.dot(clean_key.astype(np.float64),input_factor.astype(np.float64)))
    if not np.isfinite(denominator)or denominator==0:raise ValueError('Degenerate clean-key constraint')
    return ((target-base)/denominator).astype(np.float32)


def solve(model,tokenizer,request,templates,covariance,counts,steps=20,progress=None,synthetic=False):
    if templates[0]!='{}' or len(templates)!=len(set(templates)):raise ValueError('Clean template must be first; no duplicates')
    torch.manual_seed(20261027);model.eval();model.requires_grad_(False)
    module=model.transformer.h[17].mlp.c_proj;mlp=model.transformer.h[17].mlp
    examples=[example(tokenizer,t.format(request['prompt']).format(request['subject']),request['subject'],request['target'])for t in templates]
    essence=example(tokenizer,request['subject']+' is a',request['subject'])
    keys,values=[],[]
    with torch.no_grad():
        for e in examples:
            calls=[]
            def capture(mod,args,out):
                if calls:raise ValueError('Repeated c_proj in extraction')
                calls.append(True);keys.append(args[0][0,e.position].float().cpu().numpy().copy());values.append(out[0,e.position].float().cpu().numpy().copy())
            handle=module.register_forward_hook(capture)
            try:hidden(model,[e],tokenizer.eos_token_id,counts)
            finally:handle.remove()
            if len(calls)!=1:raise ValueError('Missing c_proj capture')
        native=hidden(model,[essence],tokenizer.eos_token_id,counts)
        native_kl=model.lm_head(native[0,essence.position]).float().log_softmax(-1).detach()
    keys=np.stack(keys);values=np.stack(values);mean=keys.mean(axis=0,dtype=np.float64).astype(np.float32);clean=keys[0].copy();base=values[0].copy()
    c=np.asarray(covariance,dtype=np.float64)
    if c.shape!=(len(mean),len(mean))or not np.allclose(c,c.T,atol=1e-7,rtol=0):raise ValueError('Covariance shape/symmetry')
    chol=torch.linalg.cholesky(torch.from_numpy(c));q=torch.cholesky_solve(torch.from_numpy(mean.astype(np.float64))[:,None],chol)[:,0]
    v=(q/q.norm()).numpy().astype(np.float32)
    denominator=float(np.dot(clean.astype(np.float64),v.astype(np.float64)))
    if not np.isfinite(denominator)or denominator==0:raise ValueError('Degenerate clean-key constraint')
    device=next(model.parameters()).device;initial=torch.tensor(base,device=device,dtype=torch.float32)
    if initial.norm()==0:raise ValueError('Zero clean value')
    delta=torch.zeros_like(initial,requires_grad=True);optimizer=torch.optim.Adam([delta],lr=.5,betas=(.9,.999),eps=1e-8)
    all_examples=examples+[essence];positions=[e.position for e in all_examples];history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with delta_at(mlp,positions,delta):h=hidden(model,all_examples,tokenizer.eos_token_id,counts)
        losses=[]
        for i,e in enumerate(examples):
            n=len(e.target_ids);logits=model.lm_head(h[i,len(e.ids)-n:len(e.ids)]).float()
            losses.append(F.cross_entropy(logits,torch.tensor(e.target_ids,dtype=torch.long,device=device)))
        nll=torch.stack(losses).mean()
        logq=model.lm_head(h[-1,essence.position]).float().log_softmax(-1)
        kl=.0625*torch.sum(logq.exp()*(logq-native_kl))
        decay=.5*delta.norm()/initial.norm().square()
        loss=nll+kl+decay
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():raise ValueError('Missing/nonfinite delta gradient')
        counts.update();optimizer.step()
        with torch.no_grad():
            norm=delta.norm();limit=4*initial.norm()
            if norm>limit:delta.mul_(limit/norm)
        row=dict(step=step,nll=float(nll.detach()),kl=float(kl.detach()),decay=float(decay.detach()),delta_norm=float(delta.detach().norm()))
        history.append(row)
        if progress:progress(dict(forwards=counts.forwards,gradient_steps=counts.updates,history_last=row))
    target=base+delta.detach().float().cpu().numpy();u=affine_output_factor(target,base,clean,v)
    residual=base.astype(np.float64)+u.astype(np.float64)*np.dot(v.astype(np.float64),clean.astype(np.float64))-target.astype(np.float64)
    if not np.allclose(base+u*np.dot(v,clean),target,atol=2e-5,rtol=2e-5):raise ValueError('Affine rank-one constraint failed')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('A model weight received gradients')
    return dict(U0=u[:,None],V0=v[:,None],k_star=clean,k_mean=mean,v_star=target,w_k_star=base,wk_plus_bias=base,
                context_keys=keys,context_native_values=values,bias=module.bias.detach().float().cpu().numpy().copy(),
                delta=delta.detach().float().cpu().numpy().copy(),metadata=dict(method='ROME-GPT2XL-delta-MLP-affine-clean-constraint',synthetic=synthetic,
                layer=17,module='transformer.h.17.mlp.c_proj',conv1d_weight_orientation='input_by_output; delta_W = V @ U.T',
                key_direction='C^-1 mean(context_keys), normalized to unit length',constraint_key='clean context, first shared template',
                base_includes_bias=True,steps=steps,learning_rate=.5,weight_decay=.5,clamp_norm_factor=4.,kl_factor=.0625,
                penalty='0.5 * norm(delta) / norm(clean_native_value)^2',kl_position='subject_last in subject + is a',
                kl_direction='edited || native',loss_layer='final block 47 + ln_f + tied head',seed=20261027,
                update_policy='20 actual Adam updates, no early stop, per approved A1; differs from reference loop last-iteration break',
                forwards=counts.forwards,gradient_steps=counts.updates,history=history,request=request,templates=templates,
                rewrite_input_ids=[e.ids for e in examples],target_ids=[e.target_ids for e in examples],positions=[e.position for e in examples],
                essence_input_ids=essence.ids,essence_position=essence.position,affine_constraint_max_abs=float(abs(residual).max()),
                seen_phrasings=list(dict.fromkeys([e.text for e in examples]+[essence.text,request['target']]))))


def save(result,path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/edit_cert'))
    from edit_cert.backends.lowrank import LowRank
    path=Path(path);metadata=result['metadata']
    backend=LowRank([dict(module='transformer.h.17.mlp.c_proj',layer=17,scale=1.)],
                    dict(U0=result['U0'],V0=result['V0']),label=('synthetic-'if metadata['synthetic']else'')+'Codex-P1-ROME-GPT2XL',seen_phrasings=metadata['seen_phrasings'])
    backend.save(path)
    for name,value in result.items():
        if name not in ('metadata','U0','V0'):np.save(path/f'{name}.npy',value,allow_pickle=False)
    metadata=dict(metadata,content_sha256=backend.content_sha256(),artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in path.iterdir()if p.is_file()})
    (path/'solver.json').write_text(json.dumps(metadata,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return metadata
