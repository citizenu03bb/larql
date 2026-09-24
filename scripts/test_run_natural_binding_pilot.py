import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import io
import numpy as np
import unittest

import natural_binding_pilot as ref
import run_natural_binding_pilot as run


class FrozenGateTests(unittest.TestCase):
    def setUp(self):
        self.rows=[r for r in ref.corpus() if r['split']=='discovery']
        self.jobs=[j for j in ref.ledger(ref.corpus()) if j['split']=='discovery']
        self.tokens={r['id']:dict(answer_id=1,alternative_id=2,final_position=9) for r in self.rows}
        self.records=[]
        self.events=[]
        for i,j in enumerate(self.jobs):
            rec=dict(ordinal=i,recipient=j['recipient'],arm=j['arm'],digest='same',finite=True,
                     top1_id=1,top1_logit=2.,answer_logit=2.,alternative_logit=1.,
                     applied=[[21,9]],rejected=[],max_logit_drift=0.0,
                     delta_gap=0.0,delta_target=0.0)
            if j['arm'] in ('matched','cross_construction'):
                rec.update(delta_gap=0.5,delta_target=0.3,max_logit_drift=0.5)
            self.records.append(rec)
            e=dict(ordinal=i,recipient=j['recipient'],arm=j['arm'])
            self.events.extend([dict(e,event='attempt'),dict(e,event='complete')])

    def verdict(self,n=184):
        return run.analyze('discovery',self.rows,self.tokens,self.jobs,self.records[:n],self.events[:2*n],n>40)

    def test_every_gate_required_for_confirmation(self):
        v=self.verdict()
        self.assertEqual(v['status'],'pass')
        self.assertTrue(v['confirmation_allowed'])
        self.assertEqual(len(v['rows']),16)
        self.assertEqual(self.verdict(40)['status'],'baseline_pass')
        self.assertFalse(self.verdict(40)['confirmation_allowed'])

    def test_ineligible_baseline_is_not_causal_negative(self):
        self.records[0]['top1_id']=3
        self.assertEqual(self.verdict(40)['status'],'eligibility_stop')
        self.assertEqual(self.verdict(40)['eligible_factual_prompts'],15)
        self.assertEqual(self.verdict()['status'],'mechanical_stop')

    def test_first_failed_row_stops_and_cannot_be_averaged_away(self):
        self.records[40]['delta_target']=-0.1
        self.assertEqual(self.verdict(49)['status'],'causal_negative')
        self.assertFalse(self.verdict(49)['confirmation_allowed'])
        self.assertEqual(self.verdict()['status'],'mechanical_stop')

    def test_noop_and_full_vector_self_checks(self):
        self.records[1]['digest']='different'
        self.assertEqual(self.verdict(40)['status'],'mechanical_stop')
        self.records[1]['digest']='same'
        self.records[44]['max_logit_drift']=0.01
        self.assertEqual(self.verdict(49)['status'],'mechanical_stop')

    def test_missing_forward_and_duplicate_accounting_fail_closed(self):
        self.assertEqual(self.verdict(49)['status'],'mechanical_stop')
        self.events[1]['ordinal']=1
        self.assertEqual(self.verdict()['status'],'infrastructure_stop')
        self.assertFalse(run.ledger_valid(self.jobs,self.events[:-1]))

    def test_confirmation_rejects_closed_discovery_before_any_worker(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'discovery'; out.mkdir()
            run.write_json(out/'verdict.json',dict(status='eligibility_stop',manifest_sha256='x',confirmation_allowed=False))
            with self.assertRaises(ValueError):
                run.confirmation_gate(dict(run_root=d),'x')

    def test_exclusive_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'verdict.json'
            run.write_json(p,dict(status='pass'))
            with self.assertRaises(FileExistsError):
                run.write_json(p,dict(status='changed'))


class ControllerLifecycleTests(unittest.TestCase):
    def test_eligibility_stop_persists_all_evidence_and_never_fits(self):
        # Fake worker only: no model or Rust inference is invoked.
        class FakeProcess:
            def __init__(self):
                self.stdin=io.StringIO()
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
        rows=[r for r in ref.corpus() if r['split']=='discovery']
        jobs=[j for j in ref.ledger(ref.corpus()) if j['split']=='discovery']
        state={'n':0}
        def response(process, noise):
            n=state['n']; state['n']+=1
            if n==0: return {'type':'preflight'}
            if n==1: return {'type':'ready'}
            job=jobs[n-2]
            logits=np.zeros(262144,dtype=np.float32)
            logits[3]=2.0  # Every factual answer deliberately fails eligibility.
            return dict(type='forward',ordinal=n-2,recipient=job['recipient'],arm=job['arm'],
                        logits=logits,capture=np.zeros(2560,dtype=np.float32) if job['arm']=='baseline_capture' else None,
                        applied=[],rejected=[],replacement_f32=None)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            run.write_json(root/'corpus.json',ref.corpus())
            run.write_json(root/'ledger.json',ref.ledger(ref.corpus()))
            run.write_json(root/'tokens.json',{'rows':[dict(id=r['id'],answer_id=1,alternative_id=2,final_position=9) for r in rows]})
            m=dict(run_root=str(root/'run'),corpus=str(root/'corpus.json'),ledger=str(root/'ledger.json'),
                   tokens=str(root/'tokens.json'),source_commit='fixture',binary='/not-executed',vindex='/not-read')
            with patch.object(run.subprocess,'Popen',return_value=FakeProcess()), patch.object(run,'receive',side_effect=response), patch.object(ref,'fit_rank_one',side_effect=AssertionError('must not fit')):
                with patch('sys.stdout',new=io.StringIO()):
                    self.assertEqual(run.run_stage(m,'discovery','fixture-hash',None),0)
            out=root/'run/discovery'
            v=run.read(out/'verdict.json')
            self.assertEqual(v['status'],'eligibility_stop')
            self.assertEqual(v['attempted_forwards'],40)
            self.assertEqual(v['completed_forwards'],40)
            self.assertEqual(len(list(out.glob('*.npz'))),40)
            self.assertFalse((out/'direction.json').exists())
            with self.assertRaises(FileExistsError):
                run.run_stage(m,'discovery','fixture-hash',None)


    def test_complete_positive_lifecycle_freezes_direction_and_opens_gate(self):
        class FakeProcess:
            def __init__(self): self.stdin=io.StringIO()
            def wait(self, timeout=None): return 0
            def poll(self): return 0
        rows=[r for r in ref.corpus() if r['split']=='discovery']
        by_id={r['id']:r for r in rows}
        jobs=[j for j in ref.ledger(ref.corpus()) if j['split']=='discovery']
        state={'n':0}
        def response(process, noise):
            n=state['n']; state['n']+=1
            if n==0: return {'type':'preflight'}
            if n==1: return {'type':'ready'}
            job=jobs[n-2]
            request=json.loads(process.stdin.getvalue().splitlines()[-1])
            logits=np.zeros(262144,dtype=np.float32); logits[1]=2.; logits[2]=1.
            capture=None
            if job['arm']=='baseline_capture':
                r=by_id[job['recipient']]
                capture=np.zeros(2560,dtype=np.float32)
                if r['entity'] is not None: capture[0]=(r['entity']+1)*(1-2*r['relation'])
            if job['arm'] in ('matched','cross_construction'): logits[2]+=0.5
            patched=job['arm'] in ref.ARMS
            return dict(type='forward',ordinal=n-2,recipient=job['recipient'],arm=job['arm'],logits=logits,
                        capture=capture,applied=[[21,9]] if patched else [],rejected=[],
                        replacement_f32=np.asarray(request['replacement'],dtype=np.float32) if patched else None)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            run.write_json(root/'corpus.json',ref.corpus())
            run.write_json(root/'ledger.json',ref.ledger(ref.corpus()))
            run.write_json(root/'tokens.json',{'rows':[dict(id=r['id'],answer_id=1,alternative_id=2,final_position=9) for r in rows]})
            m=dict(run_root=str(root/'run'),corpus=str(root/'corpus.json'),ledger=str(root/'ledger.json'),
                   tokens=str(root/'tokens.json'),source_commit='fixture',binary='/not-executed',vindex='/not-read')
            with patch.object(run.subprocess,'Popen',return_value=FakeProcess()), patch.object(run,'receive',side_effect=response):
                with patch('sys.stdout',new=io.StringIO()):
                    self.assertEqual(run.run_stage(m,'discovery','fixture-hash',None),0)
            out=root/'run/discovery'
            v=run.read(out/'verdict.json')
            self.assertEqual(v['status'],'pass')
            self.assertEqual(v['completed_forwards'],184)
            direction=run.confirmation_gate(m,'fixture-hash')
            self.assertEqual(direction['u'][0],1.0)
            with (out/'000-baseline_capture.npz').open('ab') as f: f.write(b'corruption')
            with self.assertRaises(ValueError): run.confirmation_gate(m,'fixture-hash')


if __name__=='__main__':
    unittest.main()
