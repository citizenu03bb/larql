"""Fixed detector/payload probes; branch to baseline only when an edit fires."""
import hashlib,json,time,sys
from pathlib import Path
R=Path(sys.argv[1])
def read(p):return json.loads(p.read_text())
def note(s):
 with (R/'decisions.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),note=s))+'\n')
 print(s,flush=True)
def used():
 p=R/'events.jsonl'
 return sum(json.loads(x)['type']=='complete' for x in p.read_text().splitlines()) if p.exists() else 0
def queue(name,job):
 assert used()+job['max_new_tokens']<=96
 p=R/'jobs'/f'{name}.pending';p.write_text(json.dumps(job,indent=2)+'\n');p.rename(p.with_suffix('.json'))
def wait(name):
 for _ in range(3500):
  p=R/'results'/f'{name}.json'
  if p.exists():return read(p)
  if (R/'STOP').exists():raise RuntimeError('session stopped')
  time.sleep(1)
 raise TimeoutError(name)
try:
 cases=read(R/'cases.json');spec=read(R/'fixed-intervention.json');binding=read(R/'parameter-binding.json');summary=[]
 for i,c in enumerate(cases):
  assert hashlib.sha256(json.dumps(read(R/'fixed-intervention.json'),sort_keys=True,separators=(',',':')).encode()).hexdigest()==binding['spec_sha256']
  name=f'{i*2:03}-auto-{c["name"]}'
  queue(name,dict(prompt_ids=c['prompt_ids'],max_new_tokens=c['max_new_tokens'],intervention=spec))
  auto=wait(name);fires=[s for s in auto['steps'] if s['triggered']]
  item=dict(case=c['name'],kind=c['kind'],auto_job=name,auto_text=auto['text'],auto_eos=auto['generated_ids'][-1] in [1,106],fire_steps=[s['step'] for s in fires],
   maximum_score=max((s['detector_score'] for s in auto['steps'] if s['detector_score'] is not None),default=None))
  note(c['name']+': '+json.dumps(item,ensure_ascii=False))
  if fires:
   step=fires[0]['step'];reserve=sum(x['max_new_tokens'] for x in cases[i+1:]);remaining=96-used()-reserve
   limit=min(c['max_new_tokens']-step,remaining)
   if limit>0:
    baseline=f'{i*2+1:03}-base-{c["name"]}'
    queue(baseline,dict(prompt_ids=c['prompt_ids'],seed_ids=auto['generated_ids'][:step],max_new_tokens=limit,
      note='Reuse only the naturally generated prefix preceding the first intervention.'))
    base=wait(baseline);item.update(baseline_job=baseline,baseline_text=base['text'],baseline_eos=base['generated_ids'][-1] in [1,106])
    note(c['name']+' baseline branch: '+base['text'])
   else:item['baseline_unavailable']='remaining budget reserved for other probes'
  else:
   item['baseline_text']=auto['text'];item['baseline_eos']=item['auto_eos'];item['baseline_basis']='Same forward: no activation modification occurred.'
  summary.append(item)
 (R/'probe-summary.json').write_text(json.dumps(dict(mode='fixed-parameter exploration',forwards=used(),budget=96,parameter_sha256=binding['spec_sha256'],cases=summary),indent=2,ensure_ascii=False)+'\n')
 note('Fixed series complete. Parameters unchanged; stop model.')
finally:(R/'STOP').touch()
