"""Adaptive queue driver; uses the single worker and its cumulative budget."""
import json,time,os
from pathlib import Path
import numpy as np
R=Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/session-20260925')
def wait(name):
 p=R/'results'/f'{name}.json'
 for _ in range(3600):
  if p.exists():return json.loads(p.read_text())
  if (R/'STOP').exists():raise RuntimeError('session stopped')
  time.sleep(1)
 raise TimeoutError(name)
def queue(name,job):
 p=R/'jobs'/f'{name}.pending'
 if p.with_suffix('.json').exists():return
 p.write_text(json.dumps(job,indent=2)+'\n');p.rename(p.with_suffix('.json'))
def note(s):
 with (R/'decisions.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),note=s))+'\n')
 print(s,flush=True)
cases=json.loads((R/'cases.json').read_text())
base=wait('000-base-hydrogen-symbol');donor=wait('001-donor-helium-prefix')
assert len(base['generated_ids'])>=8 and len(donor['generated_ids'])==8
answer_step=7;target=donor['generated_ids'][-1]
bfile=next(x['artifact'] for x in base['steps'] if x['step']==answer_step)
dfile=donor['steps'][0]['artifact']
b=np.load(R/'results'/bfile);d=np.load(R/'results'/dfile)
note(f'Baseline: {base["text"]!r}; donor target token: {target}; testing donor-minus-recipient directions at answer step 7.')
selected=None
for number,(layer,gain) in enumerate([(31,1.0),(31,2.0),(41,1.0)],2):
 delta=(d[f'layer{layer}']-b[f'layer{layer}']).astype('float32').tolist()
 spec=dict(layer=layer,gain=gain,step=answer_step,delta=delta)
 name=f'{number:03}-trial-L{layer}-gain{gain:g}'
 queue(name,dict(prompt_ids=cases[0]['prompt_ids'],seed_ids=base['generated_ids'][:answer_step],max_new_tokens=1,intervention=spec))
 result=wait(name);hit=result['generated_ids'][-1]==target
 note(f'{name}: {result["text"]!r}; donor-token hit={hit}.')
 if hit:selected=spec;break
if selected is None:
 note('Obstacle: none of the three attempted residual transfers selects the donor token.');(R/'STOP').touch();raise SystemExit(0)
(R/'selected-intervention.json').write_text(json.dumps(dict(**selected,target_id=target,recipient_artifact=bfile,donor_artifact=dfile),indent=2)+'\n')
queue('010-edit-hydrogen-symbol',dict(prompt_ids=cases[0]['prompt_ids'],seed_ids=base['generated_ids'][:7],max_new_tokens=5,intervention=selected,note='Reuse the seven naturally generated, intervention-free prefix tokens; sample every token from the intervention onward.'))
result=wait('010-edit-hydrogen-symbol');note('Free-generation edited answer: '+repr(result['text']))
for i,c in enumerate(cases[1:],1):
 baseline=f'{10+2*i:03}-base-{c["name"]}';edited=f'{11+2*i:03}-edit-{c["name"]}'
 baseline_job=dict(prompt_ids=c['prompt_ids'],max_new_tokens=12)
 if c['name']=='hydrogen-number':
  old=json.loads(Path('/home/pat/Projects/larql-research/baseline-contract-audit-v1/run-20260925/larql/results.json').read_text())
  oldcase=next(x for x in old['results'] if x['case']=='native_chat-discovery-e0-t0-r1')
  baseline_job.update(seed_ids=oldcase['generated_ids'][:7],max_new_tokens=5,note='Reuse unchanged, naturally generated seven-token prefix from the completed baseline audit.')
 queue(baseline,baseline_job)
 a=wait(baseline)
 if len(a['generated_ids'])<8:
  note(c['name']+': completed before intervention step; unchanged by construction.');continue
 queue(edited,dict(prompt_ids=c['prompt_ids'],seed_ids=a['generated_ids'][:7],max_new_tokens=5,intervention=selected,note='Reuse seven naturally generated tokens before the intervention.'))
 b=wait(edited);note(c['name']+': '+repr(a['text'])+' -> '+repr(b['text']))
note('First exploratory milestone complete; stop worker and release model memory.');(R/'STOP').touch()
