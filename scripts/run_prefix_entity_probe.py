"""Bounded follow-up of the four prepared source-context entity controls."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(destination,repo_path):
 root=Path(destination);repo=Path(repo_path);prior=root.parent/'prefix-probe-20260927';old=read(prior/'session.json');binary=old['binary']
 assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==old['binary_sha256']
 (root/'jobs').mkdir(parents=True,exist_ok=False);(root/'references').mkdir()
 session=dict(mode='exploratory larger-group entity controls',previous=str(prior),max_forwards=15,max_runtime_seconds=1500,start=time.time(),binary=binary,binary_sha256=old['binary_sha256'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),selection='smallest tested group predicting hydrogen, then recheck symbol and eight free tokens')
 write(root/'session.json',session)
 refs=[('000-reference-off','009-entity-off'),('001-reference-pair','010-entity-on')]
 for new,previous in refs:
  shutil.copy2(prior/'jobs'/f'{previous}.json',root/'jobs'/f'{new}.json')
  result=read(prior/'results'/f'{previous}.json');shutil.copy2(prior/'results'/result['steps'][0]['artifact'],root/'references'/f'{new}.npz')
  write(root/'references'/f'{new}.json',result)
 prepared=sorted((prior/'entity-controls/jobs').glob('*.json'));assert len(prepared)==4
 jobs=[]
 for i,p in enumerate(prepared,2):
  name=f'{i:03}-'+p.stem.split('-',1)[1];shutil.copy2(p,root/'jobs'/f'{name}.json');jobs.append(name)
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,'/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex','15'])
 deadline=time.monotonic()+1435
 summary=dict(status='running',references=[],entity_controls=[])
 def wait(name):
  path=root/'results'/f'{name}.json'
  while not path.exists():
   if child.poll() is not None:raise RuntimeError(f'worker exited {child.returncode}')
   if time.monotonic()>=deadline:raise TimeoutError('entity diagnostic deadline')
   time.sleep(1)
  r=read(path);print(name+': '+r['text'],flush=True);return r
 def logits(r):return np.load(root/'results'/r['steps'][0]['artifact'])['logits']
 try:
  for new,previous in refs:
   r=wait(new);equal=bool(np.array_equal(logits(r),np.load(root/'references'/f'{new}.npz')['logits']));summary['references'].append(dict(job=new,previous=previous,all_logits_bit_equal=equal));assert equal
  for name in jobs:
   r=wait(name);v=logits(r);spec=read(root/'jobs'/f'{name}.json')['intervention'];summary['entity_controls'].append(dict(job=name,positions=[x['position'] for x in spec['row_deltas']],token=r['steps'][0]['text'],hydrogen_logit=float(v[13076]),helium_logit=float(v[42404]),preserves_hydrogen=r['generated_ids'][-1]==13076))
  hits=[x for x in summary['entity_controls'] if x['preserves_hydrogen']]
  if hits:
   selected=min(hits,key=lambda x:len(x['positions']));summary['selected']=selected
   base=read(prior/'jobs/000-source-H.json');base['intervention']=read(root/'jobs'/f"{selected['job']}.json")['intervention'];write(root/'jobs/006-symbol.pending',base);(root/'jobs/006-symbol.pending').rename(root/'jobs/006-symbol.json')
   symbol=wait('006-symbol');summary['symbol']=symbol['text'];summary['symbol_is_He']=symbol['generated_ids'][-1]==2209
   if summary['symbol_is_He']:
    base['seed_ids']=symbol['generated_ids'];base['max_new_tokens']=8;write(root/'jobs/007-continuation.pending',base);(root/'jobs/007-continuation.pending').rename(root/'jobs/007-continuation.json')
    r=wait('007-continuation');summary['continuation']=r['text'];summary['continuation_job']=r['job']
   summary['status']='selected source-context candidate tested'
  else:summary['status']='none of four larger groups preserved hydrogen'
 except Exception as e:
  summary['status']='incomplete';summary['error']=repr(e);raise
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()] if (root/'events.jsonl').exists() else []
  summary.update(attempted_forwards=sum(x['type']=='attempt' for x in events),completed_forwards=sum(x['type']=='complete' for x in events),child_returncode=child.returncode)
  write(root/'summary.json',summary)
 if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
