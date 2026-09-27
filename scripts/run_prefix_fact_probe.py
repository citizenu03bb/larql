"""Locate sufficient L21 prefix rows in an aligned same-entity source pair."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def main(destination,binary,vindex):
 root=Path(destination);repo=Path(__file__).resolve().parents[1];prior=root.parent/'relation-probe-20260927'
 (root/'jobs').mkdir(parents=True,exist_ok=False)
 cases=read(prior/'source-cases.json');chosen=read(prior/'selected-intervention.json')['source_variant'];cases=[c for c in cases if c['variant']==chosen];hcase,hecase=cases
 assert len(hcase['prompt_ids'])==len(hecase['prompt_ids']) and hcase['seed_ids']==hecase['seed_ids']
 token_diff=[i for i,(a,b) in enumerate(zip(hcase['prompt_ids'],hecase['prompt_ids'])) if a!=b];assert len(token_diff)==1
 (root/'source-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
 (root/'session.json').write_text(json.dumps(dict(mode='exploratory L21 prefix localization',max_forwards=24,max_runtime_seconds=1800,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=binary,binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),previous=str(prior),start=time.time()),indent=2)+'\n')
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'24']);deadline=time.monotonic()+1750;job_index=0
 def used():
  p=root/'events.jsonl';return sum(json.loads(s)['type']=='attempt' for s in p.read_text().splitlines()) if p.exists() else 0
 def note(message):
  with (root/'decisions.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),message=message))+'\n')
  print(message,flush=True)
 def job(label,**kwargs):
  nonlocal job_index
  assert used()+kwargs['max_new_tokens']<=24
  name=f'{job_index:03}-{label}';job_index+=1;p=root/'jobs'/f'{name}.pending';p.write_text(json.dumps(kwargs,indent=2)+'\n');p.rename(p.with_suffix('.json'));p=root/'results'/f'{name}.json'
  while not p.exists():
   if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
   if time.monotonic()>deadline:raise TimeoutError('prefix diagnostic deadline')
   time.sleep(1)
  r=read(p);note(name+': '+r['text']);return r
 def save(summary):
  summary.update(forwards=used(),budget=24);(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 try:
  base=job('source-H',prompt_ids=hcase['prompt_ids'],seed_ids=hcase['seed_ids'],max_new_tokens=1)
  donor=job('source-He',prompt_ids=hecase['prompt_ids'],seed_ids=hecase['seed_ids'],max_new_tokens=1)
  assert base['generated_ids'][-1]==236814 and donor['generated_ids'][-1]==2209
  reference_checks=[]
  oldnames=read(prior/'selected-intervention.json')['source_jobs']
  for current,oldname in zip([base,donor],oldnames):
   old=read(prior/'results'/f'{oldname}.json');a=np.load(root/'results'/current['steps'][0]['artifact'])['logits'];b=np.load(prior/'results'/old['steps'][0]['artifact'])['logits'];equal=bool(np.array_equal(a,b));reference_checks.append(dict(job=current['job'],all_logits_bit_equal=equal));assert equal
  (root/'reference-checks.json').write_text(json.dumps(reference_checks,indent=2)+'\n')
  b=np.load(root/'results'/base['steps'][0]['artifact'])['prefix_layer21'];d=np.load(root/'results'/donor['steps'][0]['artifact'])['prefix_layer21'];delta=(d-b).astype('float32');assert b.shape==d.shape and b.shape[1]==2560
  norms=np.linalg.norm(delta,axis=1);support=np.flatnonzero(norms>1e-6).tolist();assert support and not np.any(delta[:token_diff[0]])
  (root/'row-norms.json').write_text(json.dumps(dict(token_difference=token_diff,support=support,norms=norms.tolist(),prefix_ids=hcase['prompt_ids']+hcase['seed_ids']),indent=2)+'\n')
  def spec(positions):return dict(layer=21,step=len(hcase['seed_ids']),gain=1.0,row_deltas=[dict(position=p,delta=delta[p].tolist()) for p in positions])
  trials=[]
  def trial(label,positions):
   r=job(label,prompt_ids=hcase['prompt_ids'],seed_ids=hcase['seed_ids'],max_new_tokens=1,intervention=spec(positions))
   v=np.load(root/'results'/r['steps'][0]['artifact'])['logits'];item=dict(job=r['job'],positions=positions,token=r['steps'][0]['text'],hit=r['generated_ids'][-1]==2209,H_logit=float(v[236814]),He_logit=float(v[2209]));trials.append(item);return r,item['hit']
  last=trial('last-row-control',[len(b)-1])
  chosen_rows=None;chosen_result=None
  for label,positions in [('convention-token',token_diff),('context-without-last',[p for p in support if p!=len(b)-1]),('all-changed-rows',support)]:
   if not positions:continue
   r,hit=trial(label,positions)
   if hit:chosen_rows=positions;chosen_result=r;break
  if chosen_rows is None:
   save(dict(status='no sufficient tested row group',trials=trials,sources=[base['job'],donor['job']]));note('No tested L21 row group produced He; stop.');return
  remaining=4
  while len(chosen_rows)>1 and remaining>0:
   cut=len(chosen_rows)//2;found=False
   for side,part in [('left',chosen_rows[:cut]),('right',chosen_rows[cut:])]:
    if remaining==0:break
    remaining-=1;r,hit=trial('split-'+side,part)
    if hit:chosen_rows=part;chosen_result=r;found=True;break
   if not found:break
  selected=spec(chosen_rows);(root/'selected-intervention.json').write_text(json.dumps(dict(spec=selected,trial_job=chosen_result['job'],sufficient_rows=chosen_rows,note='Sufficient tested group; global minimality not claimed.'),indent=2)+'\n')
  old=read(prior.parent/'fixed-probes-20260925/results/006-auto-long-sentence.json');entity=[]
  for active in [False,True]:
   args=dict(prompt_ids=hcase['prompt_ids'],seed_ids=old['generated_ids'][:12],max_new_tokens=1,note='Fixed He prefix to isolate later entity decision in source H context.')
   if active:args['intervention']=selected
   r=job('entity-'+('on' if active else 'off'),**args);v=np.load(root/'results'/r['steps'][0]['artifact'])['logits'];entity.append(dict(job=r['job'],active=active,token=r['steps'][0]['text'],hydrogen_logit=float(v[13076]),helium_logit=float(v[42404])))
  continuation=job('continuation',prompt_ids=hcase['prompt_ids'],seed_ids=chosen_result['generated_ids'],max_new_tokens=8,intervention=selected)
  save(dict(status='sufficient source-context group found',trials=trials,sources=[base['job'],donor['job']],selected_rows=chosen_rows,entity=entity,continuation_job=continuation['job'],continuation=continuation['text'],continuation_eos=continuation['generated_ids'][-1] in [1,106]));note('Prefix diagnostic complete.')
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=35)
  except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20)
 if child.returncode:raise RuntimeError(f'runner failed {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
