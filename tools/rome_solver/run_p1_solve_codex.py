#!/usr/bin/env python3
"""One approved, committed, bounded GPT-2 XL solve; no peer solve access."""
import argparse
from pathlib import Path
import time

import numpy as np

from p1_runtime import ROOT,REPO,Counts,read,sha,write,worker_guard,setup,TEMPLATE_SHA
from p1_gpt2_solver import example,solve,save

REQUESTS={
 'p1-bmw-origin-italy':dict(subject='BMW',prompt='{} is a company that was founded in the country of',target=' Italy'),
 'p1-germany-language-finnish':dict(subject='Germany',prompt='The official language of {} is',target=' Finnish'),
 'p1-gretzky-sport-baseball':dict(subject='Wayne Gretzky',prompt='{} is famous for playing the sport of',target=' baseball'),
 'p1-nintendo-origin-france':dict(subject='Nintendo',prompt='{} is a company that was founded in the country of',target=' France'),
 'p1-shakespeare-nationality-spanish':dict(subject='William Shakespeare',prompt='{} was a native of the country of',target=' Spain'),
}


def preflight(out):
    from transformers import AutoTokenizer
    from p1_runtime import MODEL
    import subprocess
    templates=read(ROOT/'templates/templates.json')['templates']
    if sha(ROOT/'templates/templates.json')!=TEMPLATE_SHA or len(templates)!=21:raise ValueError('Shared templates changed')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);requests={}
    for fact,request in REQUESTS.items():
        spec_path=ROOT/'qualification'/fact/'run/spec.json';spec=read(spec_path);s1=next(q for q in spec['questions']if q['id']=='S1')
        if s1['text']!=request['prompt'].format(request['subject'])or request['target']!=' '+spec['new']['accept'][0]:raise ValueError('Request differs from A1')
        examples=[example(tok,t.format(request['prompt']).format(request['subject']),request['subject'],request['target'])for t in templates]
        requests[fact]=dict(request=request,spec_sha256=sha(spec_path),positions=[e.position for e in examples],lengths=[len(e.ids)for e in examples],target_ids=examples[0].target_ids)
    base=read(ROOT/'covariance/preflight/preflight.json')
    covariance_status=read(ROOT/'covariance/status.json')
    if covariance_status['status']!='complete' or covariance_status['covariance_sha256']!=sha(ROOT/'covariance/C17.npy'):raise ValueError('Shared covariance must be completed before solve preflight')
    receipt=dict(base,requests=requests,templates_sha256=TEMPLATE_SHA,approval='Pat brief 40; P1 A1, 500 sequences/25 updates',
                 source_sha256={p.relative_to(REPO).as_posix():sha(p)for p in [Path(__file__).resolve(),Path(__file__).with_name('p1_runtime.py').resolve(),Path(__file__).with_name('p1_gpt2_solver.py').resolve()]},
                 expected_sequences=462,expected_updates=20,sequence_cap=500,update_cap=25,covariance_sha256=covariance_status['covariance_sha256'])
    if out.exists():raise FileExistsError(out)
    write(out,receipt);print({k:v['lengths'][-1]for k,v in requests.items()})


def worker(preflight_path,fact):
    receipt=read(preflight_path);worker_guard(receipt)
    if fact not in receipt['requests']:raise ValueError('Unregistered fact')
    out=ROOT/'solves-codex'/fact;service=ROOT/'solve-services-codex'/fact
    if out.exists()or service.exists():raise FileExistsError('No solve retry or overwrite')
    service.mkdir(parents=True);counts=Counts(500,25)
    try:
        status=read(ROOT/'covariance/status.json')
        if status['status']!='complete'or status['covariance_sha256']!=receipt['covariance_sha256']or status['covariance_sha256']!=sha(ROOT/'covariance/C17.npy'):raise ValueError('Shared C changed/incomplete')
        c=np.load(ROOT/'covariance/C17.npy',allow_pickle=False)
        if c.dtype!=np.float32 or c.shape!=(6400,6400):raise ValueError('Shared C wrong shape/dtype')
        model,tok=setup();request=receipt['requests'][fact]['request'];templates=read(ROOT/'templates/templates.json')['templates']
        result=solve(model,tok,request,templates,c,counts,progress=lambda d:write(service/'progress.json',d))
        if (counts.forwards,counts.updates)!=(462,20):raise ValueError('Unexpected exact P1 solve counters')
        result['metadata'].update(fact_id=fact,covariance_sha256=status['covariance_sha256'],templates_sha256=TEMPLATE_SHA,
                                 preflight_sha256=sha(preflight_path),source_sha256=receipt['source_sha256'],model_sha256=receipt['input_sha256'][str(Path('/mnt/models/hf-models/gpt2-xl/model.safetensors'))],attention_implementation=model.config._attn_implementation)
        meta=save(result,out)
        complete=dict(status='complete',author='Codex',fact_id=fact,forwards=counts.forwards,gradient_steps=counts.updates,
                      edit_manifest_sha256=sha(out/'solver.json'),content_sha256=meta['content_sha256'],covariance_sha256=status['covariance_sha256'],templates_sha256=TEMPLATE_SHA,seconds=time.monotonic()-counts.start)
        write(out/'status.json',complete);write(service/'status.json',complete)
    except BaseException as e:
        write(service/'status.json',dict(status='failed',forwards=counts.forwards,gradient_steps=counts.updates,error=repr(e)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['preflight','worker']);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--fact');a=p.parse_args()
    if a.stage=='preflight':preflight(a.preflight)
    else:worker(a.preflight,a.fact)


if __name__=='__main__':main()
