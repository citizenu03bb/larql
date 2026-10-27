#!/usr/bin/env python3
"""Approved P1 A1 covariance: fresh rehearsal, flock, bounded user services."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from p1_runtime import REPO,ROOT,LOCK,read,write

PY='/mnt/models/finetune/venv/bin/python'
WORKER=REPO/'tools/rome_solver/p1_covariance_codex.py'


def bounded(arguments,log,kind):
    unit=f'p1-{kind}-codex-{time.time_ns()}'
    command=['flock','-w','3600',str(LOCK),'systemd-run','--user','--wait','--collect','--pipe',f'--unit={unit}',
             '--property=MemoryHigh=22G','--property=MemoryMax=24G','--property=MemorySwapMax=8G','--property=RuntimeMaxSec=1800',
             f'--working-directory={REPO}',
             '--setenv=HF_HUB_OFFLINE=1','--setenv=TRANSFORMERS_OFFLINE=1','--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',
             '--setenv=OPENBLAS_NUM_THREADS=4','--setenv=OMP_NUM_THREADS=4','--setenv=MKL_NUM_THREADS=4',PY,str(WORKER),*arguments]
    write(log.with_suffix('.launch.json'),dict(approval='Pat brief 40, approved A1',command=command,unit=unit))
    with log.open('x')as f:
        result=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{kind} failed, no retry: {result.returncode}; see {log}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run']);p.add_argument('--approved-brief-40',action='store_true');a=p.parse_args()
    if not a.approved_brief_40:raise RuntimeError('Explicit approved scope required')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=REPO,check=True)
    subprocess.run(['git','ls-files','--error-unmatch',str(WORKER)],cwd=REPO,stdout=subprocess.DEVNULL,check=True)
    root=ROOT/'covariance';root.mkdir(exist_ok=True)
    if a.stage=='prepare':
        subprocess.run([PY,str(WORKER),'dry','--out',str(root/'dry-preparation')],cwd=REPO,check=True)
        bounded(['prepare','--out',str(root/'preflight')],root/'prepare.log','cov-preparation')
    else:
        for i in range(3):
            out=root/f'service{i}'
            if out.exists():
                if (out/'status.json').exists()and read(out/'status.json')['status']=='complete':continue
                raise RuntimeError('Existing partial/failed service; no retry')
            subprocess.run([PY,str(WORKER),'dry','--out',str(root/f'dry-service{i}')],cwd=REPO,check=True)
            bounded(['service','--out',str(out),'--preflight',str(root/'preflight/preflight.json'),'--index',str(i)],root/f'service{i}.log',f'cov-{i}')
            print(f'Covariance service {i} complete: 400 forwards',flush=True)
        subprocess.run([PY,str(WORKER),'finish','--out',str(root)],cwd=REPO,check=True)


if __name__=='__main__':main()
