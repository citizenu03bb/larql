#!/usr/bin/env python3
"""Same Codex solve on a tiny random CPU GPT-2: 462 sequences/20 updates."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import GPT2Config,GPT2LMHeadModel

from p1_runtime import ROOT,TEMPLATE_SHA,Counts,read,sha,write
from p1_gpt2_solver import solve,save
sys.path.insert(0,str(Path(__file__).parent/'tests'))
from test_p1_gpt2_solver import Tokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if sha(ROOT/'templates/templates.json')!=TEMPLATE_SHA:raise ValueError('Shared templates changed')
    torch.set_num_threads(2);torch.manual_seed(17)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=96,bos_token_id=0,eos_token_id=0,n_positions=128,n_embd=8,n_inner=16,n_layer=18,n_head=2)).eval()
    counts=Counts(500,25);templates=read(ROOT/'templates/templates.json')['templates']
    result=solve(model,Tokenizer(),dict(subject='X',prompt='{} is',target=' Y'),templates,np.eye(16,dtype=np.float32),counts,synthetic=True)
    if (counts.forwards,counts.updates)!=(462,20):raise RuntimeError('Wrong exact rehearsal counters')
    meta=save(result,a.out/'edit')
    try:counts.take(39);raise AssertionError('Sequence cap did not refuse 501')
    except RuntimeError:pass
    write(a.out/'receipt.json',dict(synthetic=True,pretrained_model_forwards=0,synthetic_sequence_forwards=counts.forwards,gradient_steps=counts.updates,
          templates_sha256=TEMPLATE_SHA,rank_one_affine_constraint_passed=True,artifact_sha256=meta['content_sha256'],solver_sha256=sha(Path(__file__).with_name('p1_gpt2_solver.py'))))


if __name__=='__main__':main()
