"""Freeze and execute the separately authorized 68-call-maximum diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
VINDEX=Path('/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex')
GGUF=Path('/mnt/models/hf-models/gemma4-e4b-gguf/gemma-4-E4B-it-Q4_K_M.gguf')
SERVER=Path('/home/pat/llama.cpp-mtp/build-cuda/bin/llama-server')
BINARY=Path('/home/pat/Projects/larql.worktree/multitoken-span-emitter-v2/target/release/examples/baseline_contract_audit')
CASES=ROOT/'docs/research-data/baseline-contract-audit-v1/cases.json'
OLD=Path('/home/pat/Projects/larql-research/natural-binding-causal-pilot-v1/run-20260924/execution-manifest.json')
PORT=18769

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    with open(p,'x') as f:
        json.dump(x,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
def event(f,x):
    f.write(json.dumps(x)+'\n'); f.flush(); os.fsync(f.fileno()); print(json.dumps(x),flush=True)
def server_args():
    return [str(SERVER),'-m',str(GGUF),'--offline','--no-warmup','--fit','off',
        '-ngl','0','--device','none','--no-op-offload','--spec-type','none',
        '-np','1','--no-cont-batching','--no-cache-prompt','--cache-ram','0',
        '--no-cache-idle-slots','-c','512','-b','512','-ub','512','-t','8','-n','8',
        '--no-mmproj','--no-ui','--host','127.0.0.1','--port',str(PORT)]
def completion_body(c):
    return dict(prompt=c['ids'],n_predict=8,temperature=0,samplers=['temperature'],
        repeat_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,seed=0,
        cache_prompt=False,return_tokens=True,n_probs=5,stream=False,ignore_eos=False)
def post(route,body):
    req=urllib.request.Request(f'http://127.0.0.1:{PORT}/{route}',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r: return json.load(r)
def freeze(out):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'commit first'
    out.mkdir(parents=True,exist_ok=False)
    files={}
    # Verify every original vindex file against the completed pilot, then bind again.
    for p,h in read(OLD)['files'].items():
        if Path(p).parent==VINDEX:
            print('verify '+str(p),flush=True); assert sha(p)==h,p; files[p]=h
    assert len(files)==17
    paths=[GGUF,SERVER,BINARY,Path(sys.executable).resolve(),CASES,ROOT/'Cargo.lock',
        ROOT/'docs/research-baseline-contract-audit-v1.md',
        ROOT/'crates/larql-lql/examples/baseline_contract_audit.rs',
        ROOT/'scripts/baseline_contract_cases.py',ROOT/'scripts/baseline_contract_static_audit.py',Path(__file__).resolve(),
        ROOT/'scripts/test_baseline_contract_audit.py',
        ROOT/'docs/research-data/baseline-contract-audit-v1/static-audit.json']
    for p in paths:
        print('hash '+str(p),flush=True); files[str(p)]=sha(p)
    write(out/'execution-manifest.json',dict(schema='larql.baseline-contract-execution.v1',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        llama_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SERVER.parents[2],text=True).strip(),
        files=files,larql_max_forwards=32,reference_max_calls=36,total_max_calls=68,
        server_args=server_args(),approval='User go ahead for input contract audit and separately bounded diagnostic',
        old_manifest_sha256=sha(OLD)))
def run(mode,out):
    m=read(out/'execution-manifest.json')
    for p,h in m['files'].items():
        # All binaries, metadata and relevant weight artifact are checked before each engine.
        if mode=='larql' and p==str(GGUF): continue
        if mode=='reference' and Path(p).parent==VINDEX and Path(p).suffix=='.bin': continue
        assert sha(p)==h, 'frozen file changed: '+p
    if mode=='larql':
        subprocess.run([str(BINARY),'run',str(VINDEX),str(CASES),str(out/'larql')],check=True)
        return
    assert read(out/'larql/results.json')['status']=='complete','LARQL must complete first'
    dest=out/'reference'; dest.mkdir(exist_ok=False)
    with open(dest/'server.log','x') as log, open(dest/'events.jsonl','x') as events:
        child=subprocess.Popen(m['server_args'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if child.poll() is not None: raise RuntimeError('reference startup failed; no retry')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=1) as r:
                        if r.status==200: break
                except (OSError,TimeoutError): pass
                time.sleep(1)
            else: raise TimeoutError('reference startup timeout')
            design=read(CASES); parity=[]
            for c in design['cases']:
                result=post('tokenize',dict(content=c['rendered'],add_special=False,parse_special=True))
                parity.append(dict(case=c['id'],result=result)); assert result['tokens']==c['ids'],'reference token mismatch'
            write(dest/'tokenizer-parity.json',parity)
            results=[]
            for ordinal,c in enumerate(design['cases']):
                body=completion_body(c)
                event(events,dict(type='request_attempt',ordinal=ordinal,case=c['id'],max_model_calls=9,body=body))
                result=post('completion',body)
                write(dest/f'{ordinal:03}-response.json',result)
                assert 0<=result['tokens_predicted']<=8
                assert result['tokens_evaluated']==len(c['ids'])
                assert len(result['tokens'])<=8
                item=dict(case=c['id'],response=result)
                results.append(item)
                event(events,dict(type='request_complete',ordinal=ordinal,case=c['id'],tokens=result['tokens'],text=result['content'],stop=result['stop_type']))
            write(dest/'results.json',dict(status='complete',max_calls_accounted=36,results=results))
        finally:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','larql','reference']);p.add_argument('output',type=Path);a=p.parse_args()
    if a.mode=='freeze': freeze(a.output)
    else: run(a.mode,a.output)
