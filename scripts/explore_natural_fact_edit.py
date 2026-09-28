"""Short adaptive experiment queue. No sealed gates; hard 128-forward session cap."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
def run(root,binary,vindex,max_forwards=128):
 max_forwards=int(max_forwards);assert 1<=max_forwards<=128
 root=Path(root);(root/'results').mkdir(exist_ok=False)
 tok=Tokenizer.from_file(str(Path(vindex)/'tokenizer.json'))
 event=(root/'events.jsonl').open('x');noise=(root/'worker.log').open('x')
 child=subprocess.Popen([binary,vindex],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=noise,text=True,bufsize=1)
 count=0
 def log(x):event.write(json.dumps(x)+'\n');event.flush();os.fsync(event.fileno());print(json.dumps(x),flush=True)
 def receive():
  for line in child.stdout:
   if line.startswith('EXP\t'):return json.loads(line[4:])
   noise.write(line);noise.flush()
  raise RuntimeError('worker exited')
 def forward(job,step,ids,intervention):
  nonlocal count
  if count>=max_forwards:raise RuntimeError('session forward budget exhausted')
  request=dict(ids=ids,layer=None,position=None,delta=None,gain=0,detector=None)
  if intervention:request.update(intervention)
  log(dict(type='attempt',ordinal=count,job=job,step=step,ids=ids,layer=request['layer'],position=request['position'],gain=request['gain'],automatic=request['detector'] is not None))
  child.stdin.write(json.dumps(request)+'\n');child.stdin.flush();r=receive()
  assert r['ordinal']==count and r['type']=='forward'
  logits=np.asarray(r.pop('logits'),dtype='<f4');assert logits.shape==(262144,) and np.isfinite(logits).all()
  captures={f'layer{layer}':np.asarray(row,dtype='<f4') for layer,row in r.pop('captures')}
  captures.update({f'prefix_layer{layer}':np.asarray(rows,dtype='<f4') for layer,rows in r.pop('prefix_captures',[])})
  if r.get('kv22') is not None:
   captures['kv22_keys']=np.asarray(r['kv22'][0],dtype='<f4');captures['kv22_values']=np.asarray(r['kv22'][1],dtype='<f4')
  filename=f'{count:03}.npz';np.savez_compressed(root/'results'/filename,logits=logits,**captures)
  top=int(logits.argmax());topids=np.argsort(logits)[-5:][::-1].tolist()
  record=dict(type='complete',ordinal=count,job=job,step=step,artifact=filename,top1=top,
   text=tok.decode([top],skip_special_tokens=False),top5=[dict(id=i,text=tok.decode([i],skip_special_tokens=False),logit=float(logits[i])) for i in topids],applied=r['applied'],applied_position=r['applied_position'],detector_score=r['detector_score'],triggered=r['triggered'])
  if 'applied_positions' in r:record['applied_positions']=r['applied_positions']
  for key in ['residual_replaced','kv_replaced']:
   if key in r:record[key]=r[key]
  count+=1;log(record);return top,record
 try:
  assert receive()['type']=='ready';log(dict(type='ready',budget=max_forwards))
  done=set()
  while count<max_forwards:
   if (root/'STOP').exists():break
   pending=sorted(p for p in (root/'jobs').glob('*.json') if p.name not in done)
   if not pending:time.sleep(1);continue
   path=pending[0];job=json.loads(path.read_text());name=path.stem
   generated=list(job.get('seed_ids',[]));ids=job['prompt_ids']+generated;steps=[]
   max_new=job.get('max_new_tokens',1);assert 1<=max_new<=32
   latched=None
   for k in range(max_new):
    if count>=max_forwards:break
    step=len(generated);inter=None
    spec=job.get('intervention')
    if spec and 'boundary' in spec and step>=spec['step']:
     inter=dict(layer=22,gain=1,boundary=spec['boundary'])
    elif spec and 'row_deltas' in spec and step>=spec['step']:
     inter=dict(layer=spec['layer'],gain=spec['gain'],row_deltas=spec['row_deltas'])
    elif spec and spec.get('detector'):
     inter=dict(layer=spec['layer'],position=latched,delta=spec['delta'],gain=spec['gain'],detector=spec['detector'])
    elif spec and step>=spec['step']:
     inter=dict(layer=spec['layer'],position=len(job['prompt_ids'])+spec['step']-1,delta=spec['delta'],gain=spec['gain'])
    top,record=forward(name,step,ids,inter);steps.append(record);generated.append(top);ids=ids+[top]
    if record['triggered']:latched=record['applied_position']
    if top in [1,106]:break
   result=dict(job=name,generated_ids=generated,text=tok.decode(generated,skip_special_tokens=False),new_forwards=len(steps),steps=steps,budget_exhausted=count>=max_forwards)
   write(root/'results'/f'{name}.json',result);log(dict(type='job_complete',job=name,text=result['text'],session_forwards=count));done.add(path.name)
  log(dict(type='session_complete',forwards=count))
 finally:
  child.stdin.close()
  try:child.wait(timeout=15)
  except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=15)
  child.stdout.close()
  event.close();noise.close()
if __name__=='__main__':run(*sys.argv[1:])
