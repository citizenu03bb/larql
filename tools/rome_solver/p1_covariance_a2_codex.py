#!/usr/bin/env python3
"""Pat-approved A2: reuse frozen sample, six 200-forward parts after A1 fails."""
import argparse
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from p1_runtime import REPO,ROOT,LOCK,Counts,read,write,sha,worker_guard,setup

PY='/mnt/models/finetune/venv/bin/python'
HERE=Path(__file__).resolve()
BASE=ROOT/'covariance'


def rehearsal(out):
    from rome_solver.covariance import SecondMoment
    from transformers.pytorch_utils import Conv1D
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);data=np.random.default_rng(17).normal(size=(1200,4,8));combined=SecondMoment(8)
    for i in range(6):
        stat=SecondMoment(8);count=Counts(200);module=Conv1D(4,8).double();calls=[]
        def pre(mod,args):calls.append(True);stat.add(args[0][0].detach().numpy())
        h=module.register_forward_pre_hook(pre)
        with torch.no_grad():
            for row in data[i*200:(i+1)*200]:
                calls.clear();count.take();module(torch.tensor(row[None]));assert len(calls)==1
        h.remove();assert count.forwards==200
        try:count.take();raise AssertionError('201st sequence accepted')
        except RuntimeError:pass
        combined.sum+=stat.sum;combined.count+=stat.count
    expected=SecondMoment(8)
    for row in data:expected.add(row)
    actual,_=combined.finish();target,_=expected.finish();np.testing.assert_allclose(actual,target,atol=1e-7,rtol=1e-6)
    write(out/'receipt.json',dict(synthetic=True,pretrained_model_forwards=0,parts=6,sequences_per_part=200,single_ridge=True,passed=True))


def worker(index,out,preflight):
    from rome_solver.covariance import SecondMoment
    receipt=read(preflight);worker_guard(receipt)
    if out.exists()or index not in range(6):raise ValueError('New A2 part required; no retry')
    if sha(receipt['windows_path'])!=receipt['windows_sha256']:raise ValueError('Frozen sample changed')
    out.mkdir(parents=True);count=Counts(200)
    try:
        windows=np.load(receipt['windows_path'],allow_pickle=False)[index*200:(index+1)*200];model,tok=setup();stat=SecondMoment(6400);calls=[]
        def capture(mod,args):
            if calls:raise RuntimeError('Repeated projection capture')
            calls.append(True);stat.add(args[0][0].detach().float().cpu().numpy())
        h=model.transformer.h[17].mlp.c_proj.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for window in windows:
                    calls.clear();count.take();hidden=model.transformer(input_ids=torch.tensor(window[None],dtype=torch.long,device='cuda'),use_cache=False).last_hidden_state
                    if len(calls)!=1 or not torch.isfinite(hidden).all():raise RuntimeError('Capture/nonfinite error')
                    write(out/'progress.json',dict(forwards=count.forwards,keys=stat.count,seconds=time.monotonic()-count.start))
        finally:h.remove()
        if (count.forwards,stat.count)!=(200,204800):raise RuntimeError('Wrong A2 counters')
        np.save(out/'sum.npy',stat.sum,allow_pickle=False)
        write(out/'status.json',dict(status='complete',amendment='A2',forwards=200,keys=stat.count,sum_sha256=sha(out/'sum.npy'),preflight_sha256=sha(preflight),seconds=time.monotonic()-count.start,attention_implementation=model.config._attn_implementation))
    except BaseException as e:
        write(out/'status.json',dict(status='failed',forwards=count.forwards,error=repr(e)));raise


def finish(preflight):
    from rome_solver.covariance import SecondMoment
    if (BASE/'C17.npy').exists()or(BASE/'status.json').exists():raise FileExistsError('Shared covariance already published')
    total=SecondMoment(6400);statuses=[]
    for i in range(6):
        p=BASE/f'parts-a2/service{i}';s=read(p/'status.json')
        if s['status']!='complete'or(s['forwards'],s['keys'])!=(200,204800)or s['preflight_sha256']!=sha(preflight)or sha(p/'sum.npy')!=s['sum_sha256']:raise RuntimeError('Invalid A2 part')
        total.sum+=np.load(p/'sum.npy',allow_pickle=False);total.count+=s['keys'];statuses.append(s)
    c,meta=total.finish();np.linalg.cholesky(c.astype(np.float64));pending=BASE/'C17.pending.npy';np.save(pending,c,allow_pickle=False);pending.replace(BASE/'C17.npy')
    discarded=read(BASE/'service0/status.json')['forwards']
    write(BASE/'status.json',dict(status='complete',author='Codex',amendment='A2 approved by Pat',covariance_sha256=sha(BASE/'C17.npy'),forwards=1200,
          discarded_A1_forwards=discarded,total_sequence_forwards=1200+discarded,gradient_steps=0,shape=list(c.shape),dtype=str(c.dtype),
          layer=17,module='transformer.h.17.mlp.c_proj',sample=read(BASE/'preflight/sample.json'),moment=meta,partial_statuses=statuses,preflight_sha256=sha(preflight)))
    print('P1 covariance published '+sha(BASE/'C17.npy'),flush=True)


def launch():
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=REPO,check=True)
    while not (BASE/'service0/status.json').exists():
        if (BASE/'status.json').exists():raise RuntimeError('A1 completed; conditional A2 is unnecessary')
        time.sleep(10)
    failed=read(BASE/'service0/status.json')
    if failed['status']!='failed':raise RuntimeError('Conditional A2 requires failed A1 service')
    if 'budget/deadline exhausted' not in failed.get('error',''):raise RuntimeError('A1 failed for another reason; timing amendment does not authorize retry')
    preflight=BASE/'a2-preflight.json'
    if not preflight.exists():
        original=read(BASE/'preflight/preflight.json');original.update(amendment='A2',approval='Pat async approval: Approuver A2 : 6 × 200',
            windows_path=str(BASE/'preflight/windows.npy'),services=6,forwards_per_service=200,discarded_A1_status_sha256=sha(BASE/'service0/status.json'),
            source_sha256={p.relative_to(REPO).as_posix():sha(p)for p in [HERE,HERE.with_name('p1_runtime.py'),REPO/'tools/rome_solver/rome_solver/covariance.py']})
        write(preflight,original)
    for i in range(6):
        out=BASE/f'parts-a2/service{i}'
        if out.exists():
            if (out/'status.json').exists()and read(out/'status.json')['status']=='complete':continue
            raise RuntimeError('Existing failed/partial A2 part; no retry')
        rehearsal(BASE/f'dry-a2-service{i}')
        from run_registered_solves import host_snapshot
        write(BASE/f'a2-service{i}-host-before.json',host_snapshot())
        unit=f'p1-cov-a2-{i}-codex-{time.time_ns()}'
        command=['flock','-w','3600',str(LOCK),'systemd-run','--user','--wait','--collect','--pipe',f'--unit={unit}',
          '--property=MemoryHigh=22G','--property=MemoryMax=24G','--property=MemorySwapMax=8G','--property=RuntimeMaxSec=1800',f'--working-directory={REPO}',
          '--setenv=HF_HUB_OFFLINE=1','--setenv=TRANSFORMERS_OFFLINE=1','--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',
          '--setenv=OPENBLAS_NUM_THREADS=4','--setenv=OMP_NUM_THREADS=4','--setenv=MKL_NUM_THREADS=4',PY,str(HERE),'worker','--index',str(i),'--out',str(out),'--preflight',str(preflight)]
        write(BASE/f'a2-service{i}.launch.json',dict(unit=unit,command=command,approval='Pat A2'))
        with (BASE/f'a2-service{i}.log').open('x')as f:result=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('A2 service failed; stop, no retry')
        print(f'A2 covariance part {i+1}/6 complete',flush=True)
    finish(preflight)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['rehearsal','worker','launch']);p.add_argument('--approved-a2',action='store_true');p.add_argument('--index',type=int);p.add_argument('--out',type=Path);p.add_argument('--preflight',type=Path);a=p.parse_args()
    if a.stage=='rehearsal':rehearsal(a.out)
    elif a.stage=='worker':worker(a.index,a.out,a.preflight)
    else:
        if not a.approved_a2:raise RuntimeError('Explicit A2 approval required')
        launch()


if __name__=='__main__':main()
