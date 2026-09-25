"""Model-free controller integration test using a synthetic worker."""
import json,subprocess,tempfile,unittest
from pathlib import Path
import explore_natural_fact_edit as runner
class Budget(unittest.TestCase):
 def test_stops_mid_job_at_session_cap(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'jobs').mkdir()
   worker=root/'worker.py'
   worker.write_text('''#!/usr/bin/python3
import json,sys
print('EXP\\t'+json.dumps({'type':'ready'}),flush=True)
for i,line in enumerate(sys.stdin):
 print('EXP\\t'+json.dumps(dict(type='forward',ordinal=i,logits=[1.0]+[0.0]*262143,captures=[],applied=False,applied_position=None,detector_score=None,triggered=False)),flush=True)
''');worker.chmod(0o755)
   for i in range(2):(root/'jobs'/f'{i}.json').write_text(json.dumps(dict(prompt_ids=[2],max_new_tokens=2)))
   runner.run(root,str(worker),'/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex',3)
   events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
   self.assertEqual(sum(x['type']=='attempt' for x in events),3)
   self.assertEqual(json.loads((root/'results/1.json').read_text())['new_forwards'],1)
   self.assertTrue(json.loads((root/'results/1.json').read_text())['budget_exhausted'])
if __name__=='__main__':unittest.main()
