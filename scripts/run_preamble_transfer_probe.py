"""One bounded alignment hypothesis: six identical answer-preamble tokens at L21."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main(destination,repo_path):
 root=Path(destination);repo=Path(repo_path);parent=root.parent;prefix=parent/'prefix-probe-20260927';relation=parent/'relation-probe-20260927';entity=parent/'prefix-entity-probe-20260927';old=read(prefix/'session.json');binary=old['binary']
 assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()==old['binary_sha256']
 (root/'jobs').mkdir(parents=True,exist_ok=False);(root/'references').mkdir()
 h=read(prefix/'jobs/000-source-H.json');he=read(prefix/'jobs/001-source-He.json');target=read(relation/'jobs/006-replay-long-symbol.json')
 assert h['seed_ids']==he['seed_ids']==target['seed_ids'] and len(h['seed_ids'])==6
 assert len(h['prompt_ids'])==len(he['prompt_ids'])==48 and len(target['prompt_ids'])==31
 def reference(folder,oldname,label):
  r=read(folder/'results'/f'{oldname}.json');p=folder/'results'/r['steps'][0]['artifact'];shutil.copy2(p,root/'references'/f'{label}.npz');shutil.copy2(folder/'jobs'/f'{oldname}.json',root/'references'/f'{label}-job.json');write(root/'references'/f'{label}-result.json',r);return np.load(p)
 b=reference(prefix,'000-source-H','source-H')['prefix_layer21'];d=reference(prefix,'001-source-He','source-He')['prefix_layer21'];delta=d-b
 reference(relation,'006-replay-long-symbol','natural-H');reference(entity,'006-symbol','positive-four')
 assert b.shape==d.shape==(54,2560)
 mapping=[dict(source_position=48+i,target_position=31+i,token=token) for i,token in enumerate(h['seed_ids'])]
 write(root/'alignment.json',dict(method='identical answer-preamble token IDs, offset from prompt end',mapping=mapping,source_prompt_ids=h['prompt_ids'],donor_prompt_ids=he['prompt_ids'],target_prompt_ids=target['prompt_ids']))
 def spec(offset):return dict(layer=21,step=6,gain=1.0,row_deltas=[dict(position=offset+i,delta=delta[48+i].tolist()) for i in range(6)])
 source_spec=spec(48);target_spec=spec(31);write(root/'interventions.json',dict(source=source_spec,natural=target_spec))
 session=dict(mode='exploratory six-token preamble transfer',max_forwards=15,max_runtime_seconds=1500,start=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=binary,binary_sha256=old['binary_sha256'],donor_source=str(prefix),selection='fixed gain 1, no row search; stop on natural symbol miss; otherwise entity on/off and eight free continuation tokens')
 write(root/'session.json',session)
 refs=[('000-source-reference',h,'source-H'),('001-natural-reference',target,'natural-H'),('002-positive-four',read(entity/'jobs/006-symbol.json'),'positive-four')]
 for name,j,_ in refs:write(root/'jobs'/f'{name}.json',j)
 summary=dict(status='running',references=[],trials=[]);child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,'/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex','15']);deadline=time.monotonic()+1435
 def wait(name):
  path=root/'results'/f'{name}.json'
  while not path.exists():
   if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
   if time.monotonic()>=deadline:raise TimeoutError('preamble transfer deadline')
   time.sleep(1)
  r=read(path);print(name+': '+r['text'],flush=True);return r
 def job(name,j):
  count=sum(json.loads(x)['type']=='attempt' for x in (root/'events.jsonl').read_text().splitlines());assert count+j['max_new_tokens']<=15
  write(root/'jobs'/f'{name}.pending',j);(root/'jobs'/f'{name}.pending').rename(root/'jobs'/f'{name}.json');return wait(name)
 def logits(r):return np.load(root/'results'/r['steps'][0]['artifact'])['logits']
 try:
  for name,j,label in refs:
   r=wait(name);equal=bool(np.array_equal(logits(r),np.load(root/'references'/f'{label}.npz')['logits']));summary['references'].append(dict(job=name,reference=label,all_logits_bit_equal=equal));assert equal
  for name,j,specification in [('003-source-preamble',h,source_spec),('004-natural-preamble',target,target_spec)]:
   r=job(name,dict(prompt_ids=j['prompt_ids'],seed_ids=j['seed_ids'],max_new_tokens=1,intervention=specification));v=logits(r);summary['trials'].append(dict(job=name,token=r['steps'][0]['text'],He_minus_H=float(v[2209]-v[236814]),symbol_is_He=r['generated_ids'][-1]==2209))
  symbol=r
  if not summary['trials'][-1]['symbol_is_He']:summary['status']='natural symbol transfer failed at tested alignment and gain'
  else:
   summary['entity']=[]
   for active in [False,True]:
    j=read(relation/'jobs/010-entity-he-off.json');j.pop('intervention',None)
    if active:j['intervention']=target_spec
    r=job('005-entity-off' if not active else '006-entity-on',j);v=logits(r);summary['entity'].append(dict(active=active,token=r['steps'][0]['text'],hydrogen_minus_helium=float(v[13076]-v[42404])))
   r=job('007-continuation',dict(prompt_ids=target['prompt_ids'],seed_ids=symbol['generated_ids'],max_new_tokens=8,intervention=target_spec));summary['continuation']=r['text'];summary['status']='natural symbol transfer and continuation tested'
 except Exception as e:
  summary.update(status='incomplete',error=repr(e));raise
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  p=root/'events.jsonl';e=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
  summary.update(attempted_forwards=sum(x['type']=='attempt' for x in e),completed_forwards=sum(x['type']=='complete' for x in e),child_returncode=child.returncode);write(root/'summary.json',summary)
 if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
