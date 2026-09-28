"""Fixed six-pass residual localization with the positive KV22 state held constant."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main(destination,repo_path):
 root=Path(destination);repo=Path(repo_path);prior=root.parent/'kv-boundary-20260928';old=read(prior/'session.json');binary=old['binary'];assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==old['binary_sha256']
 (root/'jobs').mkdir(parents=True,exist_ok=False);(root/'references').mkdir()
 def copy_reference(name,label):
  j=read(prior/'jobs'/f'{name}.json');r=read(prior/'results'/f'{name}.json');shutil.copy2(prior/'results'/r['steps'][0]['artifact'],root/'references'/f'{label}.npz');write(root/'references'/f'{label}-job.json',j);write(root/'references'/f'{label}-result.json',r);return j,np.load(root/'references'/f'{label}.npz')
 basejob,base=copy_reference('000-baseline','baseline');posjob,pos=copy_reference('001-positive','positive');assert basejob['prompt_ids']==posjob['prompt_ids'] and basejob['seed_ids']==posjob['seed_ids']
 assert len(basejob['prompt_ids'])==48 and len(basejob['seed_ids'])==6
 groups=[('002-prompt',list(range(15,48))),('003-answer',list(range(48,54))),('004-original-four',list(range(15,19))),('005-remaining-prompt',list(range(19,48)))]
 changed=np.flatnonzero(np.any(base['prefix_layer22']!=pos['prefix_layer22'],axis=1)).tolist();assert changed==list(range(15,54))
 assert groups[0][1]+groups[1][1]==changed and groups[2][1]+groups[3][1]==groups[0][1]
 write(root/'groups.json',dict(tested_groups=[dict(job=n,positions=p) for n,p in groups],fixed_kv_changed_positions=[15,16,17,18],contrast='replace selected L22 rows with successful source-state rows; all other residual rows baseline; positive KV22 constant'))
 write(root/'session.json',dict(mode='exploratory fixed residual groups with positive KV22',max_forwards=6,max_runtime_seconds=900,start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=binary,binary_sha256=old['binary_sha256'],previous=str(prior),scope='same source prompt and supplied answer preamble, symbol only; no global-minimum claim'))
 refs=[]
 for name,previous in [('000-reference-kv-only','004-kv-only'),('001-reference-combined','005-combined')]:
  j,_=copy_reference(previous,name);write(root/'jobs'/f'{name}.json',j);refs.append(name)
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,'/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex','6']);deadline=time.monotonic()+840;summary=dict(status='running',references=[],conditions=[])
 def wait(name):
  p=root/'results'/f'{name}.json'
  while not p.exists():
   if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
   if time.monotonic()>deadline:raise TimeoutError('residual-group diagnostic deadline')
   time.sleep(1)
  r=read(p);print(name+': '+r['text'],flush=True);return r
 def measure(name,r,positions):
  z=np.load(root/'results'/r['steps'][0]['artifact']);v=z['logits'];summary['conditions'].append(dict(job=name,positions=positions,token=r['steps'][0]['text'],He_minus_H=float(v[2209]-v[236814])));return z
 try:
  for name,positions in zip(refs,[[],changed]):
   r=wait(name);z=measure(name,r,positions);reference=np.load(root/'references'/f'{name}.npz');equal={k:bool(np.array_equal(z[k],reference[k])) for k in reference.files};summary['references'].append(dict(job=name,bit_equal_arrays=equal));assert all(equal.values())
  assert [x['token'] for x in summary['conditions']]==['H','He']
  for name,positions in groups:
   residual=base['prefix_layer22'].copy();residual[positions]=pos['prefix_layer22'][positions]
   boundary=dict(residual=residual.tolist(),kv=dict(keys=pos['kv22_keys'].tolist(),values=pos['kv22_values'].tolist()))
   job=dict(prompt_ids=basejob['prompt_ids'],seed_ids=basejob['seed_ids'],max_new_tokens=1,intervention=dict(layer=22,step=6,gain=1,boundary=boundary));write(root/'jobs'/f'{name}.pending',job);(root/'jobs'/f'{name}.pending').rename(root/'jobs'/f'{name}.json');r=wait(name);measure(name,r,positions)
  summary['status']='completed fixed residual-group diagnostic'
 except Exception as e:
  summary.update(status='incomplete',error=repr(e));raise
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  p=root/'events.jsonl';events=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else [];summary.update(attempted_forwards=sum(x['type']=='attempt' for x in events),completed_forwards=sum(x['type']=='complete' for x in events),child_returncode=child.returncode);write(root/'summary.json',summary)
 if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
