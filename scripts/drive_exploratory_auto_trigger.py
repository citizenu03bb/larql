"""Adaptive automatic-trigger exploration; all jobs share one capped worker."""
import json,time
from pathlib import Path
import numpy as np
R=Path('/home/pat/Projects/larql-research/exploratory-natural-fact-edit/auto-trigger-20260925')
OLD=R.parent/'session-20260925'
def read(p):return json.loads(p.read_text())
def note(s):
 with (R/'decisions.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),note=s))+'\n')
 print(s,flush=True)
def wait(name):
 for _ in range(3500):
  p=R/'results'/f'{name}.json'
  if p.exists():return read(p)
  if (R/'STOP').exists():raise RuntimeError('stopped')
  time.sleep(1)
 raise TimeoutError(name)
def used():return sum(json.loads(x)['type']=='complete' for x in (R/'events.jsonl').read_text().splitlines())
def queue(name,job):
 if used()+job['max_new_tokens']>128:raise RuntimeError('remaining budget insufficient for requested job')
 p=R/'jobs'/f'{name}.pending';p.write_text(json.dumps(job,indent=2)+'\n');p.rename(p.with_suffix('.json'))
def outcome(d):return dict(text=d['text'],fires=[dict(step=s['step'],score=s['detector_score'],position=s['applied_position']) for s in d['steps'] if s['triggered']])
def fit_with_shifted(baseline):
 rows=[];labels=[];names=[]
 for entry in read(R/'calibration-v1.json')['rows']:
  h=np.load(entry['artifact'])['layer31'].astype(float);rows.append(h/np.linalg.norm(h));labels.append(entry['positive']);names.append([entry['job'],entry['step']])
 for s in baseline['steps']:
  h=np.load(R/'results'/s['artifact'])['layer31'].astype(float);rows.append(h/np.linalg.norm(h));labels.append(s['top1']==236814);names.append([baseline['job'],s['step']])
 for name in ['001-auto-number','002-auto-helium']:
  for s in read(R/'results'/f'{name}.json')['steps']:
   if not s['applied']:
    h=np.load(R/'results'/s['artifact'])['layer31'].astype(float);rows.append(h/np.linalg.norm(h));labels.append(False);names.append([name,s['step']])
 x=np.array(rows);labels=np.array(labels);y=np.where(labels,1.0,-1.0)
 w=x.T@np.linalg.solve(x@x.T+0.01*np.eye(len(x)),y);w/=np.linalg.norm(w)
 scores=x@w;lo=float(scores[labels].min());hi=float(scores[~labels].max());assert lo>hi
 threshold=(lo+hi)/2
 spec=read(R/'detector-v1.json');spec['detector']=dict(key=w.astype('float32').tolist(),threshold=threshold)
 (R/'detector-v2.json').write_text(json.dumps(spec,indent=2)+'\n')
 (R/'calibration-v2.json').write_text(json.dumps(dict(method='ridge on normalized residuals, lambda 0.01; adaptive development',positive_min=lo,negative_max=hi,threshold=threshold,rows=[dict(name=n,positive=bool(p),score=float(s)) for n,p,s in zip(names,labels,scores)]),indent=2)+'\n')
 return spec
try:
 cases=read(R/'cases.json');spec=read(R/'detector-v1.json')
 for name in ['000-auto-canonical','001-auto-number','002-auto-helium']:
  d=wait(name);note(name+': '+json.dumps(outcome(d)))
 baseline=wait('003-base-shifted');note('Shifted baseline: '+baseline['text'])
 assert any(s['top1']==236814 for s in baseline['steps']),'baseline did not emit H'
 queue('004-auto-shifted-v1',dict(prompt_ids=cases[-1]['prompt_ids'],max_new_tokens=20,intervention=spec))
 result=wait('004-auto-shifted-v1');note('Shifted v1: '+json.dumps(outcome(result)))
 version='v1'
 if not any(s['triggered'] for s in result['steps']) or 2209 not in result['generated_ids']:
  spec=fit_with_shifted(baseline);version='v2'
  note('Adaptation: fit a normalized-residual linear direction including the newly observed answer format; do not change the payload.')
  queue('005-auto-shifted-v2',dict(prompt_ids=cases[-1]['prompt_ids'],max_new_tokens=20,intervention=spec))
  result=wait('005-auto-shifted-v2');note('Shifted v2: '+json.dumps(outcome(result)))
  if any(s['triggered'] for s in result['steps']) and 2209 not in result['generated_ids']:
   spec['gain']=2.0;version='v2-gain2';note('Trigger fires but H persists: try payload gain 2 within this session.')
   queue('006-auto-shifted-gain2',dict(prompt_ids=cases[-1]['prompt_ids'],max_new_tokens=20,intervention=spec))
   result=wait('006-auto-shifted-gain2');note('Shifted gain2: '+json.dumps(outcome(result)))
  for name,c in [('010-auto-canonical-final',cases[0]),('011-auto-number-final',cases[2]),('012-auto-helium-final',cases[1])]:
   if used()+12>128:note('Budget stop before '+name);break
   queue(name,dict(prompt_ids=c['prompt_ids'],max_new_tokens=12,intervention=spec));d=wait(name);note(name+': '+json.dumps(outcome(d)))
 (R/'selected-detector.json').write_text(json.dumps(dict(version=version,spec=spec,shifted_result=result['job']),indent=2)+'\n')
 note('Automatic-trigger exploration complete; release model. New material used adaptively, not as independent validation.')
finally:(R/'STOP').touch()
