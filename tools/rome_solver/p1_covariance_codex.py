#!/usr/bin/env python3
"""P1 A1: fixed WikiText-103 sample and three bounded covariance workers.

No Claude solve imports. Reuses Codex's unchanged fixed_windows and
SecondMoment estimator. Partial float64 sums are combined before ONE ridge.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch

from p1_runtime import REPO,ROOT,MODEL,TEMPLATE_SHA,Counts,read,sha,write,worker_guard,setup


def verify_manifest(folder):
    result={}
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(None,1);name=name.strip().lstrip('*')
        if sha(folder/name)!=expected:raise ValueError('Input checksum mismatch '+name)
        result[str(folder/name)]=expected
    return result


def prepare(out):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from rome_solver.covariance import fixed_windows
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    dataset=Path('/home/pat/Projects/larql-research/datasets/wikitext-103-raw-v1')
    hashes={**verify_manifest(MODEL),**verify_manifest(dataset)}
    if sha(ROOT/'templates/templates.json')!=TEMPLATE_SHA:raise ValueError('Shared templates hash mismatch')
    records=[];article=[]
    for path in sorted(dataset.glob('train-*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=['text']):
            for line in batch.column(0).to_pylist():
                stripped=line.strip()
                if re.fullmatch(r'= [^=].* =',stripped):
                    if article and ''.join(article).strip():records.append((str(len(records)),''.join(article)))
                    article=[]
                article.append(line if line.endswith('\n')else line+'\n')
    if article and ''.join(article).strip():records.append((str(len(records)),''.join(article)))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    windows,metadata=fixed_windows({'wiki':records},tok,quotas={'wiki':1200},window=1024,seed=20261026)
    np.save(out/'windows.npy',windows,allow_pickle=False);write(out/'sample.json',metadata)
    receipt=dict(approval='Pat brief 40; P1 amendment A1',model_stats={p.name:[p.stat().st_size,p.stat().st_mtime_ns]for p in MODEL.iterdir()if p.is_file()},
        input_sha256=hashes,templates_sha256=TEMPLATE_SHA,windows_sha256=sha(out/'windows.npy'),sample_sha256=sha(out/'sample.json'),
        source_sha256={p.relative_to(REPO).as_posix():sha(p)for p in [Path(__file__).resolve(),Path(__file__).with_name('p1_runtime.py').resolve(),REPO/'tools/rome_solver/rome_solver/covariance.py']},
        articles=len(records),windows=1200,tokens_per_window=1024,positions='all',keys=1228800,ridge=1e-4,services=3,forwards_per_service=400)
    write(out/'preflight.json',receipt);print(json.dumps({'articles':len(records),'windows':len(windows),'windows_sha256':receipt['windows_sha256']}),flush=True)


def service(preflight,index,out):
    from rome_solver.covariance import SecondMoment
    receipt=read(preflight);worker_guard(receipt)
    if index not in (0,1,2)or out.exists():raise ValueError('Invalid/new service output required')
    out.mkdir(parents=True);counts=Counts(400)
    try:
        if sha(preflight.parent/'windows.npy')!=receipt['windows_sha256']:raise ValueError('Sample changed')
        windows=np.load(preflight.parent/'windows.npy',allow_pickle=False)[index*400:(index+1)*400]
        model,tok=setup();stat=SecondMoment(6400);module=model.transformer.h[17].mlp.c_proj;calls=[]
        def capture(mod,args):
            if calls:raise RuntimeError('Projection captured twice')
            calls.append(True);stat.add(args[0][0].detach().float().cpu().numpy())
        handle=module.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for j,window in enumerate(windows):
                    calls.clear();counts.take()
                    hidden=model.transformer(input_ids=torch.tensor(window[None],dtype=torch.long,device='cuda'),use_cache=False).last_hidden_state
                    if len(calls)!=1 or not torch.isfinite(hidden).all():raise RuntimeError('Capture/nonfinite failure')
                    write(out/'progress.json',dict(forwards=counts.forwards,keys=stat.count,seconds=time.monotonic()-counts.start))
        finally:handle.remove()
        if counts.forwards!=400 or stat.count!=409600:raise RuntimeError('Wrong final covariance count')
        np.save(out/'sum.npy',stat.sum,allow_pickle=False)
        write(out/'status.json',dict(status='complete',forwards=400,keys=stat.count,sum_sha256=sha(out/'sum.npy'),preflight_sha256=sha(preflight),seconds=time.monotonic()-counts.start,attention_implementation=model.config._attn_implementation))
    except BaseException as e:
        write(out/'status.json',dict(status='failed',forwards=counts.forwards,error=repr(e)));raise


def finish(root):
    from rome_solver.covariance import SecondMoment
    if (root/'C17.npy').exists()or(root/'status.json').exists():raise FileExistsError('Covariance already published')
    stat=SecondMoment(6400);statuses=[]
    for i in range(3):
        folder=root/f'service{i}';s=read(folder/'status.json')
        if s['status']!='complete'or s['forwards']!=400 or s['keys']!=409600 or s['preflight_sha256']!=sha(root/'preflight/preflight.json') or sha(folder/'sum.npy')!=s['sum_sha256']:raise ValueError('Failed/changed partial covariance')
        stat.sum+=np.load(folder/'sum.npy',allow_pickle=False);stat.count+=s['keys'];statuses.append(s)
    c,meta=stat.finish(1e-4)
    np.linalg.cholesky(c.astype(np.float64))
    temp=root/'C17.pending.npy';np.save(temp,c,allow_pickle=False);temp.replace(root/'C17.npy')
    write(root/'status.json',dict(status='complete',author='Codex',covariance_sha256=sha(root/'C17.npy'),forwards=1200,gradient_steps=0,
        layer=17,module='transformer.h.17.mlp.c_proj',shape=list(c.shape),dtype=str(c.dtype),sample=read(root/'preflight/sample.json'),moment=meta,
        partial_statuses=statuses,preflight_sha256=sha(root/'preflight/preflight.json')))
    print(json.dumps({'status':'complete','covariance_sha256':sha(root/'C17.npy'),'forwards':1200}),flush=True)


def dry(out):
    from rome_solver.covariance import SecondMoment
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);data=np.random.default_rng(17).normal(size=(1200,4,8));parts=[]
    for i in range(3):
        stat=SecondMoment(8);counts=Counts(400)
        from transformers.pytorch_utils import Conv1D
        module=Conv1D(4,8).double();calls=[]
        def capture(mod,args):calls.append(True);stat.add(args[0][0].detach().numpy())
        handle=module.register_forward_pre_hook(capture)
        with torch.no_grad():
            for row in data[i*400:(i+1)*400]:
                calls.clear();counts.take();module(torch.tensor(row[None]));assert len(calls)==1
        handle.remove();assert counts.forwards==400
        try:counts.take();raise AssertionError('Counter did not refuse 401')
        except RuntimeError:pass
        parts.append(stat)
    combined=SecondMoment(8)
    for stat in parts:combined.sum+=stat.sum;combined.count+=stat.count
    actual,meta=combined.finish();all_at_once=SecondMoment(8)
    for row in data:all_at_once.add(row)
    expected,_=all_at_once.finish();np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
    write(out/'receipt.json',dict(synthetic=True,model_forwards=0,three_services=3,sequence_counts=[400]*3,single_ridge=True,keys=combined.count,passed=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','service','finish','dry']);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path);p.add_argument('--index',type=int);a=p.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='service':service(a.preflight,a.index,a.out)
    elif a.stage=='finish':finish(a.out)
    else:dry(a.out)


if __name__=='__main__':main()
