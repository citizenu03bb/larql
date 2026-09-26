"""Bounded exploratory L21 transfer plus same-prefix continuation contrasts."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(destination,binary,vindex):
 root=Path(destination);repo=Path(__file__).resolve().parents[1];previous=root.parent/'fixed-probes-20260925';first=root.parent/'session-20260925'
 (root/'jobs').mkdir(parents=True,exist_ok=False)
 session=dict(mode='exploratory L21 transfer',max_forwards=24,max_runtime_seconds=1500,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=str(binary),binary_sha256=sha(Path(binary)),start=time.time(),previous=str(previous),first=str(first))
 (root/'session.json').write_text(json.dumps(session,indent=2)+'\n')
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'24'])
 deadline=time.monotonic()+1450
 def used():
  p=root/'events.jsonl'
  return sum(json.loads(x)['type']=='attempt' for x in p.read_text().splitlines()) if p.exists() else 0
 def note(message):
  with (root/'decisions.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),message=message))+'\n')
  print(message,flush=True)
 def job(name,**kwargs):
  assert used()+kwargs.get('max_new_tokens',1)<=24
  p=root/'jobs'/f'{name}.pending';p.write_text(json.dumps(kwargs,indent=2)+'\n');p.rename(p.with_suffix('.json'))
  result=root/'results'/f'{name}.json'
  while not result.exists():
   if child.poll() is not None:raise RuntimeError(f'runner exited {child.returncode}')
   if time.monotonic()>deadline:raise TimeoutError('L21 deadline')
   time.sleep(1)
  result=read(result);note(name+': '+result['text']);return result
 try:
  initial=read(first/'jobs/002-trial-L31-gain1.json');donor=read(first/'jobs/001-donor-helium-prefix.json')
  recipient=job('000-capture-recipient',prompt_ids=initial['prompt_ids'],seed_ids=initial['seed_ids'],max_new_tokens=1)
  donor_result=job('001-capture-donor',prompt_ids=donor['prompt_ids'],seed_ids=donor['seed_ids'],max_new_tokens=1)
  assert recipient['generated_ids'][-1]==236814 and donor_result['generated_ids'][-1]==2209
  b=np.load(root/'results'/recipient['steps'][0]['artifact'])['layer21'];d=np.load(root/'results'/donor_result['steps'][0]['artifact'])['layer21'];delta=(d-b).astype('float32');assert np.isfinite(delta).all()
  prior_auto=read(previous/'results/006-auto-long-sentence.json');prior_base=read(previous/'results/007-base-long-sentence.json');prompt=read(previous/'jobs/006-auto-long-sentence.json')['prompt_ids'];seed=prior_base['generated_ids'][:6]
  baseline=job('002-replay-long-symbol',prompt_ids=prompt,seed_ids=seed,max_new_tokens=1)
  trials=[];selected=None;trial=None
  for n,gain in enumerate([1.0,2.0],3):
   spec=dict(layer=21,step=6,gain=gain,delta=delta.tolist())
   trial=job(f'{n:03}-trial-gain{gain:g}',prompt_ids=prompt,seed_ids=seed,max_new_tokens=1,intervention=spec)
   hit=trial['generated_ids'][-1]==2209;trials.append(dict(gain=gain,job=trial['job'],symbol=trial['steps'][0]['text'],hit=hit));selected=spec
   if hit:break
  selection=dict(spec=selected,trials=trials,symbol_hit=trials[-1]['hit'],delta_norm=float(np.linalg.norm(delta)),recipient_norm=float(np.linalg.norm(b)),direction='canonical donor-minus-recipient post-L21',detector=None)
  (root/'selected-intervention.json').write_text(json.dumps(selection,indent=2)+'\n')
  note('Selected L21 gain '+str(selected['gain'])+'; symbol hit='+str(selection['symbol_hit']))
  matrix=[]
  for n,(label,ref,active) in enumerate([('h-off',prior_base,False),('h-on',prior_base,True),('he-off',prior_auto,False),('he-on',prior_auto,True)],10):
   args=dict(prompt_ids=prompt,seed_ids=ref['generated_ids'][:12],max_new_tokens=1,note='Diagnostic fixed prefix; not a from-prompt generation claim.')
   if active:args['intervention']=selected
   r=job(f'{n:03}-entity-{label}',**args);v=np.load(root/'results'/r['steps'][0]['artifact'])['logits']
   matrix.append(dict(condition=label,job=r['job'],token=r['steps'][0]['text'],hydrogen_logit=float(v[13076]),helium_logit=float(v[42404]),helium_minus_hydrogen=float(v[42404]-v[13076])))
  continuation=job('020-free-continuation',prompt_ids=prompt,seed_ids=trial['generated_ids'],max_new_tokens=8,intervention=selected,note='Reuse the selected trial symbol, then freely generate eight tokens with intervention retained.')
  summary=dict(forwards=used(),budget=24,trials=trials,selected_gain=selected['gain'],symbol_hit=selection['symbol_hit'],matrix=matrix,continuation=continuation['text'],continuation_eos=continuation['generated_ids'][-1] in [1,106])
  (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');note('L21 probe complete.')
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=35)
  except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20)
 if child.returncode:raise RuntimeError(f'runner failed: {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
