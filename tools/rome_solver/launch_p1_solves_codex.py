#!/usr/bin/env python3
"""Five approved independent Codex solves, serial, fresh-model services."""
import argparse
import subprocess
import time

from p1_runtime import ROOT,REPO,LOCK,read,sha,write
from run_p1_solve_codex import REQUESTS
from run_registered_solves import host_snapshot

PY='/mnt/models/finetune/venv/bin/python'
WORKER=REPO/'tools/rome_solver/run_p1_solve_codex.py'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--approved-brief-40',action='store_true');a=p.parse_args()
    if not a.approved_brief_40:raise RuntimeError('Approved scope required')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=REPO,check=True)
    subprocess.run(['git','ls-files','--error-unmatch',str(WORKER)],cwd=REPO,stdout=subprocess.DEVNULL,check=True)
    preflight=ROOT/'solves-codex-preflight.json'
    if not preflight.exists():subprocess.run([PY,str(WORKER),'preflight','--preflight',str(preflight)],cwd=REPO,check=True)
    logs=ROOT/'solve-launches-codex';logs.mkdir(exist_ok=True)
    finished=[]
    for fact in REQUESTS:
        out=ROOT/'solves-codex'/fact
        if out.exists():
            status=read(out/'status.json')
            if status['status']!='complete'or status['edit_manifest_sha256']!=sha(out/'solver.json'):raise RuntimeError('Invalid existing edit; no retry')
            finished.append(status);continue
        if (ROOT/'solve-services-codex'/fact).exists():raise RuntimeError('Existing failed/active solve; no retry')
        write(logs/(fact+'-host-before.json'),host_snapshot())
        subprocess.run([PY,str(REPO/'tools/rome_solver/dry_p1_solve_codex.py'),'--out',str(ROOT/'solve-rehearsals-codex'/fact)],cwd=REPO,check=True)
        unit=f'p1-solve-codex-{time.time_ns()}'
        command=['flock','-w','3600',str(LOCK),'systemd-run','--user','--wait','--collect','--pipe',f'--unit={unit}',
          '--property=MemoryHigh=22G','--property=MemoryMax=24G','--property=MemorySwapMax=8G','--property=RuntimeMaxSec=1800',f'--working-directory={REPO}',
          '--setenv=HF_HUB_OFFLINE=1','--setenv=TRANSFORMERS_OFFLINE=1','--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',
          '--setenv=OPENBLAS_NUM_THREADS=4','--setenv=OMP_NUM_THREADS=4','--setenv=MKL_NUM_THREADS=4',PY,str(WORKER),'worker','--preflight',str(preflight),'--fact',fact]
        write(logs/(fact+'.launch.json'),dict(unit=unit,command=command,approval='Pat brief 40'))
        with (logs/(fact+'.log')).open('x')as f:result=subprocess.run(command,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('P1 solve failed; stop, no retry')
        status=read(out/'status.json');finished.append(status)
        write(ROOT/'solves-codex-index.json',dict(author='Codex',complete=False,cases=finished,model_sequence_forwards=sum(s['forwards']for s in finished),gradient_steps=sum(s['gradient_steps']for s in finished)))
        print(f'{fact}: complete, 462 sequences, 20 updates',flush=True)
    write(ROOT/'solves-codex-index.json',dict(author='Codex',complete=True,cases=finished,model_sequence_forwards=sum(s['forwards']for s in finished),gradient_steps=sum(s['gradient_steps']for s in finished)))


if __name__=='__main__':main()
