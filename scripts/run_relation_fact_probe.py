"""Same-entity counterfactual relation contrast at L21; bounded diagnostic."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(destination,binary,vindex):
 root=Path(destination);repo=Path(__file__).resolve().parents[1];previous=root.parent/'fixed-probes-20260925';first=root.parent/'session-20260925'
 (root/'jobs').mkdir(parents=True,exist_ok=False)
 session=dict(mode='exploratory same-entity relation transfer',max_forwards=24,max_runtime_seconds=1800,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),binary=str(binary),binary_sha256=sha(Path(binary)),start=time.time(),previous=str(previous),first=str(first))
 (root/'session.json').write_text(json.dumps(session,indent=2)+'\n')
 child=subprocess.Popen([sys.executable,'-u',str(repo/'scripts/explore_natural_fact_edit.py'),str(root),binary,vindex,'24'])
 deadline=time.monotonic()+1750
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
   if time.monotonic()>deadline:raise TimeoutError('relation probe deadline')
   time.sleep(1)
  result=read(result);note(name+': '+result['text']);return result
 try:
  prior_auto=read(previous/'results/006-auto-long-sentence.json');prior_base=read(previous/'results/007-base-long-sentence.json');prompt=read(previous/'jobs/006-auto-long-sentence.json')['prompt_ids'];seed=prior_base['generated_ids'][:6]
  question=next(c['question'] for c in read(previous/'cases.json') if c['name']=='long-sentence')
  tokenizer=Tokenizer.from_file(str(Path(vindex)/'tokenizer.json'));template=ImmutableSandboxedEnvironment().from_string((Path(vindex)/'chat_template.jinja').read_text())
  source_cases=[];pair=None
  for variant,wording in enumerate([
   "In this fictional chemistry exercise, hydrogen's symbol is {symbol}. Use this convention. ",
   "Use this stipulated mapping: hydrogen -> {symbol}. Treat it as the definition for this exercise, even if it differs from ordinary chemistry. "
  ]):
   captures=[];pair_cases=[]
   for offset,symbol in enumerate(['H','He']):
    text=wording.format(symbol=symbol)+question
    rendered=template.render(messages=[dict(role='user',content=text)],bos_token='<bos>',add_generation_prompt=True,enable_thinking=False,tools=None)
    ids=tokenizer.encode(rendered,add_special_tokens=False).ids
    case=dict(variant=variant,symbol=symbol,question=text,prompt_ids=ids,seed_ids=seed)
    pair_cases.append(case);source_cases.append(case)
    captures.append(job(f'{variant*2+offset:03}-source-{variant}-{symbol}',prompt_ids=ids,seed_ids=seed,max_new_tokens=1,note='Explicit fictional convention; matched entity and answer prefix.'))
   (root/'source-cases.json').write_text(json.dumps(source_cases,indent=2)+'\n')
   ids1,ids2=[c['prompt_ids'] for c in pair_cases]
   note('Source pair token lengths: '+str([len(ids1),len(ids2)]))
   if captures[0]['generated_ids'][-1]==236814 and captures[1]['generated_ids'][-1]==2209:
    pair=(captures,pair_cases);break
  if pair is None:
   (root/'summary.json').write_text(json.dumps(dict(status='source contrast unavailable',forwards=used(),budget=24),indent=2)+'\n')
   note('Neither source pair predicts H versus He; stop before transfer.');return
  captures,pair_cases=pair
  b=np.load(root/'results'/captures[0]['steps'][0]['artifact'])['layer21'];d=np.load(root/'results'/captures[1]['steps'][0]['artifact'])['layer21'];delta=(d-b).astype('float32');assert np.isfinite(delta).all()
  donor_entity=job('005-source-He-entity',prompt_ids=pair_cases[1]['prompt_ids'],seed_ids=prior_auto['generated_ids'][:12],max_new_tokens=1,note='Fixed He prefix to check the stipulated subject; not a complete free response.')
  baseline=job('006-replay-long-symbol',prompt_ids=prompt,seed_ids=seed,max_new_tokens=1)
  trials=[];selected=None;trial=None
  for n,gain in enumerate([1.0,2.0],7):
   spec=dict(layer=21,step=6,gain=gain,delta=delta.tolist())
   trial=job(f'{n:03}-trial-gain{gain:g}',prompt_ids=prompt,seed_ids=seed,max_new_tokens=1,intervention=spec)
   hit=trial['generated_ids'][-1]==2209;trials.append(dict(gain=gain,job=trial['job'],symbol=trial['steps'][0]['text'],hit=hit));selected=spec
   if hit:break
  selection=dict(spec=selected,trials=trials,symbol_hit=trials[-1]['hit'],delta_norm=float(np.linalg.norm(delta)),recipient_norm=float(np.linalg.norm(b)),direction='same-entity fictional He-minus-H post-L21',source_jobs=[r['job'] for r in captures],source_variant=pair_cases[0]['variant'],source_entity_token=donor_entity['steps'][0]['text'],detector=None)
  (root/'selected-intervention.json').write_text(json.dumps(selection,indent=2)+'\n')
  note('Selected L21 gain '+str(selected['gain'])+'; symbol hit='+str(selection['symbol_hit']))
  matrix=[]
  for n,(label,ref,active) in enumerate([('he-off',prior_auto,False),('he-on',prior_auto,True)],10):
   args=dict(prompt_ids=prompt,seed_ids=ref['generated_ids'][:12],max_new_tokens=1,note='Diagnostic fixed prefix; not a from-prompt generation claim.')
   if active:args['intervention']=selected
   r=job(f'{n:03}-entity-{label}',**args);v=np.load(root/'results'/r['steps'][0]['artifact'])['logits']
   matrix.append(dict(condition=label,job=r['job'],token=r['steps'][0]['text'],hydrogen_logit=float(v[13076]),helium_logit=float(v[42404]),helium_minus_hydrogen=float(v[42404]-v[13076])))
  continuation=job('020-free-continuation',prompt_ids=prompt,seed_ids=trial['generated_ids'],max_new_tokens=8,intervention=selected,note='Reuse the selected trial symbol, then freely generate eight tokens with intervention retained.')
  summary=dict(status='complete',source_jobs=selection['source_jobs'],source_variant=selection['source_variant'],source_entity_token=selection['source_entity_token'],forwards=used(),budget=24,trials=trials,selected_gain=selected['gain'],symbol_hit=selection['symbol_hit'],matrix=matrix,continuation=continuation['text'],continuation_eos=continuation['generated_ids'][-1] in [1,106])
  (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');note('Same-entity relation probe complete.')
 finally:
  (root/'STOP').touch()
  try:child.wait(timeout=35)
  except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20)
 if child.returncode:raise RuntimeError(f'runner failed: {child.returncode}')
if __name__=='__main__':main(*sys.argv[1:])
