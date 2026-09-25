import json
from pathlib import Path
import tempfile
import unittest
import run_baseline_contract_audit as r
class Contract(unittest.TestCase):
    def test_cases_use_only_two_spent_prompts(self):
        d=r.read(r.CASES)
        self.assertEqual(len(d['cases']),4)
        self.assertEqual({c['source_row'] for c in d['cases']},{'discovery-e0-t0-r0','discovery-e0-t0-r1'})
        self.assertEqual([len(c['ids']) for c in d['cases']],[9,9,16,16])
        self.assertEqual(sum(c['original_logit_digest'] is not None for c in d['cases']),2)
    def test_reference_no_hidden_budget(self):
        args=r.server_args()
        for flag in ['--no-warmup','--offline','--no-cache-prompt']: self.assertIn(flag,args)
        self.assertEqual(args[args.index('--spec-type')+1],'none')
        for c in r.read(r.CASES)['cases']:
            b=r.completion_body(c)
            self.assertEqual(b['prompt'],c['ids']);self.assertEqual(b['n_predict'],8)
            self.assertEqual(b['samplers'],['temperature']); self.assertEqual(b['temperature'],0)
    def test_durable_evidence_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';r.write(p,{'first':True})
            with self.assertRaises(FileExistsError):r.write(p,{'second':True})
            self.assertEqual(r.read(p),{'first':True})
if __name__=='__main__':unittest.main()
