"""Bounded local GPT-2 XL runtime, shared by Codex's P1 workers only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

REPO = Path(__file__).resolve().parents[2]
MODEL = Path('/mnt/models/hf-models/gpt2-xl')
ROOT = Path('/home/pat/Projects/larql-research/phase-b-p1-v2-20261027')
LOCK = Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock')
TEMPLATE_SHA = 'b7dce8e1fa21e362e63e4ea45fc5a25030cce7a61781dcdd55f0c0216afebf15'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


class Counts:
    def __init__(self,cap,steps=0):self.cap,self.step_cap,self.forwards,self.updates,self.start=cap,steps,0,0,time.monotonic()
    def take(self,sequences=1):
        if self.forwards+sequences>self.cap or time.monotonic()-self.start>1740:raise RuntimeError('P1 budget/deadline exhausted')
        self.forwards+=sequences
    def update(self):
        if self.updates>=self.step_cap:raise RuntimeError('P1 update budget exhausted')
        self.updates+=1


def worker_guard(receipt):
    if not os.environ.get('INVOCATION_ID'):raise RuntimeError('Real worker requires bounded systemd service')
    probe=subprocess.run(['flock','-n',str(LOCK),'true'],capture_output=True)
    if probe.returncode!=1:raise RuntimeError('Shared model lock is not held')
    cg=next(line.split('::',1)[1]for line in Path('/proc/self/cgroup').read_text().splitlines()if line.startswith('0::'))
    group=Path('/sys/fs/cgroup')/cg.lstrip('/')
    for name,limit in [('memory.high',22*1024**3),('memory.max',24*1024**3),('memory.swap.max',8*1024**3)]:
        value=(group/name).read_text().strip()
        if value=='max'or int(value)>limit:raise RuntimeError('Missing cgroup bound '+name)
    for name,expected in receipt['source_sha256'].items():
        if sha(REPO/name)!=expected:raise RuntimeError('Worker source changed '+name)
    for name,expected in receipt['model_stats'].items():
        stat=(MODEL/name).stat()
        if [stat.st_size,stat.st_mtime_ns]!=expected:raise RuntimeError('Model file changed '+name)
    if sha(ROOT/'templates/templates.json')!=TEMPLATE_SHA:raise RuntimeError('Shared templates changed')


def setup():
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(4);torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    model.requires_grad_(False)
    if model.config.n_layer!=48 or model.config.n_embd!=1600:raise RuntimeError('Wrong GPT-2 model')
    if tuple(model.transformer.h[17].mlp.c_proj.weight.shape)!=(6400,1600):raise RuntimeError('Wrong Conv1D orientation')
    return model,tokenizer
