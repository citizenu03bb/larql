"""Six forward passes separate the end-of-L22 residual and reused K/V state."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main(destination,repo_path,binary):
 root=Path(destination);repo=Path(repo_path);old=root.parent/'l22-effect-transfer-20260928'
 (root/'jobs').mkdir(parents=True,exist_ok=False);(root/'references').mkdir()
 write(root/'session.json',dict(mode='exploratory L22 residual x cache boundary',max_forwards=6,max_runtime_seconds=900,start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=binary,binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),previous=str(old),scope='same source-H prompt, symbol decision only; no natural-question or entity claim'))
 refs=[]
 for name,previous in [('000-baseline','000-source-off'),('001-positive','001-source-L21-on')]:
  j=read(old/'jobs'/f'{previous}.json');r=read(old/'results'/f'{previous}.json');write(root/'jobs'/f'{name}.json',j);write(root/'references'/f'{name}-job.json',j);write(root/'references'/f'{name}-result.json',r);shutil.copy2(old/'results'/r['steps'][0]['artifact'],root/'references'/f'{name}.npz');refs.append(name)
 baseline_job=read(root/'jobs/000-baseline.json');positive_job=read(root/'jobs/001-positive.json')
 assert baseline_job['prompt_ids']==positive_job['prompt_ids'] and baseline_job['seed_ids']==positive_job['seed_ids']
 assert len(baseline_job['prompt_ids'])==48 and len(baseline_job['seed_ids'])==6
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,'/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex','6']);deadline=time.monotonic()+840;summary=dict(status='running',references=[],conditions=[])
 def wait(name):
  p=root/'results'/f'{name}.json'
  while not p.exists():
   if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
   if time.monotonic()>deadline:raise TimeoutError('KV boundary diagnostic deadline')
   time.sleep(1)
  r=read(p);print(name+': '+r['text'],flush=True);return r
 def arrays(r):return np.load(root/'results'/r['steps'][0]['artifact'])
 def measure(name,r):
  a=arrays(r);v=a['logits'];item=dict(job=name,token=r['steps'][0]['text'],He_minus_H=float(v[2209]-v[236814]));summary['conditions'].append(item);return a
 try:
  current=[]
  for name in refs:
   r=wait(name);new=measure(name,r);reference=np.load(root/'references'/f'{name}.npz');equal={k:bool(np.array_equal(new[k],reference[k])) for k in reference.files};summary['references'].append(dict(job=name,bit_equal_arrays=equal));assert all(equal.values());current.append(new)
  base,pos=current;assert summary['conditions'][0]['token']=='H' and summary['conditions'][1]['token']=='He'
  j=read(root/'jobs/000-baseline.json')
  for name,residual,kv in [('002-self-replay',base,base),('003-residual-only',pos,None),('004-kv-only',None,pos),('005-combined',pos,pos)]:
   boundary=dict(residual=None if residual is None else residual['prefix_layer22'].tolist(),kv=None if kv is None else dict(keys=kv['kv22_keys'].tolist(),values=kv['kv22_values'].tolist()))
   job=dict(prompt_ids=j['prompt_ids'],seed_ids=j['seed_ids'],max_new_tokens=1,intervention=dict(layer=22,step=6,gain=1,boundary=boundary));write(root/'jobs'/f'{name}.pending',job);(root/'jobs'/f'{name}.pending').rename(root/'jobs'/f'{name}.json')
   r=wait(name);a=measure(name,r)
   if name=='002-self-replay':
    equal={k:bool(np.array_equal(a[k],base[k])) for k in base.files};summary['self_replay_bit_equal']=equal;assert all(equal.values())
   if name=='005-combined':summary['combined_logits_bit_equal_positive']=bool(np.array_equal(a['logits'],pos['logits']))
  summary['status']='completed six-condition source diagnostic'
 except Exception as e:
  summary.update(status='incomplete',error=repr(e));raise
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  p=root/'events.jsonl';e=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else [];summary.update(attempted_forwards=sum(x['type']=='attempt' for x in e),completed_forwards=sum(x['type']=='complete' for x in e),child_returncode=child.returncode);write(root/'summary.json',summary)
 if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
