"""Transfer the L22 answer-preamble effect induced by the successful L21 patch."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(destination,repo_path,binary):
 root=Path(destination);repo=Path(repo_path);parent=root.parent;prefix=parent/'prefix-probe-20260927';entity=parent/'prefix-entity-probe-20260927';relation=parent/'relation-probe-20260927'
 (root/'jobs').mkdir(parents=True,exist_ok=False);(root/'references').mkdir()
 def reference(folder,name,label):
  j=read(folder/'jobs'/f'{name}.json');r=read(folder/'results'/f'{name}.json');shutil.copy2(folder/'results'/r['steps'][0]['artifact'],root/'references'/f'{label}.npz');write(root/'references'/f'{label}-job.json',j);write(root/'references'/f'{label}-result.json',r);return j
 h=reference(prefix,'000-source-H','source-H');positive=reference(entity,'006-symbol','source-positive');target=reference(relation,'006-replay-long-symbol','natural-H')
 assert h['prompt_ids']==positive['prompt_ids'] and h['seed_ids']==positive['seed_ids']==target['seed_ids'] and len(h['seed_ids'])==6
 assert len(h['prompt_ids'])==48 and len(target['prompt_ids'])==31
 write(root/'alignment.json',dict(method='identical six-token answer preamble',mapping=[dict(source_position=48+i,target_position=31+i,token=x) for i,x in enumerate(h['seed_ids'])],source_prompt_ids=h['prompt_ids'],target_prompt_ids=target['prompt_ids'],contrast='same source-H input, successful L21 patch on minus off; captured after L22'))
 write(root/'session.json',dict(max_forwards=15,max_runtime_seconds=1500,start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=binary,binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),stop_rule='Stop on source L22 symbol miss, then on natural symbol miss; otherwise two entity probes plus eight free tokens.',gain=1,patch_layer=22))
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,'/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex','15']);deadline=time.monotonic()+1435;summary=dict(status='running',references=[],trials=[])
 def wait(name):
  p=root/'results'/f'{name}.json'
  while not p.exists():
   if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
   if time.monotonic()>deadline:raise TimeoutError('L22 effect-transfer deadline')
   time.sleep(1)
  r=read(p);print(name+': '+r['text'],flush=True);return r
 def job(name,j):
  p=root/'events.jsonl';used=sum(json.loads(x)['type']=='attempt' for x in p.read_text().splitlines()) if p.exists() else 0;assert used+j['max_new_tokens']<=15
  write(root/'jobs'/f'{name}.pending',j);(root/'jobs'/f'{name}.pending').rename(root/'jobs'/f'{name}.json');return wait(name)
 def arrays(r):return np.load(root/'results'/r['steps'][0]['artifact'])
 def replay(name,j,label):
  r=job(name,j);new=arrays(r);old=np.load(root/'references'/f'{label}.npz');equal={k:bool(np.array_equal(new[k],old[k])) for k in old.files};summary['references'].append(dict(job=name,reference=label,bit_equal_arrays=equal));assert all(equal.values());return r
 def trial(name,j,spec):
  r=job(name,dict(prompt_ids=j['prompt_ids'],seed_ids=j['seed_ids'],max_new_tokens=1,intervention=spec));v=arrays(r)['logits'];summary['trials'].append(dict(job=name,token=r['steps'][0]['text'],He_minus_H=float(v[2209]-v[236814]),symbol_is_He=r['generated_ids'][-1]==2209));return r
 try:
  base=replay('000-source-off',h,'source-H');on=replay('001-source-L21-on',positive,'source-positive');assert base['generated_ids'][-1]==236814 and on['generated_ids'][-1]==2209
  b=arrays(base)['prefix_layer22'];d=arrays(on)['prefix_layer22'];assert b.shape==d.shape==(54,2560);delta=d-b
  np.savez_compressed(root/'induced-L22-effect.npz',delta=delta,source_off=b,source_on=d)
  def spec(offset):return dict(layer=22,step=6,gain=1.0,row_deltas=[dict(position=offset+i,delta=delta[48+i].tolist()) for i in range(6)])
  source_spec=spec(48);natural_spec=spec(31);write(root/'interventions.json',dict(source=source_spec,natural=natural_spec,preamble_row_norms=np.linalg.norm(delta[48:54],axis=1).tolist()))
  trial('002-source-L22-replay',h,source_spec)
  if not summary['trials'][-1]['symbol_is_He']:summary['status']='source L22 preamble replay failed; natural transfer not attempted'
  else:
   replay('003-natural-reference',target,'natural-H');symbol=trial('004-natural-L22-transfer',target,natural_spec)
   if not summary['trials'][-1]['symbol_is_He']:summary['status']='natural L22 transfer failed'
   else:
    summary['entity']=[]
    for active in [False,True]:
     j=read(relation/'jobs/010-entity-he-off.json');j.pop('intervention',None)
     if active:j['intervention']=natural_spec
     r=job('005-entity-off' if not active else '006-entity-on',j);v=arrays(r)['logits'];summary['entity'].append(dict(active=active,token=r['steps'][0]['text'],hydrogen_minus_helium=float(v[13076]-v[42404])))
    r=job('007-continuation',dict(prompt_ids=target['prompt_ids'],seed_ids=symbol['generated_ids'],max_new_tokens=8,intervention=natural_spec));summary.update(status='natural L22 symbol transfer and continuation tested',continuation=r['text'])
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
