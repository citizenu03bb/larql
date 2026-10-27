"""Tiny randomly initialized CPU GPT-2: no pretrained weights or service."""
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import GPT2Config,GPT2LMHeadModel

sys.path.insert(0,str(Path(__file__).parents[1]))
from p1_gpt2_solver import solve,save,example,affine_output_factor
from p1_runtime import Counts


class Tokenizer:
    eos_token_id=0
    def encode(self,text,**kwargs):return [1+ord(c)%89 for c in text]
    def __call__(self,text,**kwargs):return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1)for i in range(len(text))])


class PortTests(unittest.TestCase):
    def test_conv1d_bias_multitoken_loss_and_frozen_weight_roundtrip(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=96,bos_token_id=0,eos_token_id=0,n_positions=128,n_embd=8,n_inner=16,n_layer=18,n_head=2)).eval()
        before={k:v.detach().clone()for k,v in model.state_dict().items()}
        counts=Counts(500,25)
        result=solve(model,Tokenizer(),dict(subject='X',prompt='{} is',target=' Y'),['{}','A {}','B {}'],np.eye(16,dtype=np.float32),counts,steps=2,synthetic=True)
        self.assertEqual((counts.forwards,counts.updates),(12,2))
        self.assertTrue(all(torch.equal(v,before[k])for k,v in model.state_dict().items()))
        self.assertEqual(result['U0'].shape,(8,1));self.assertEqual(result['V0'].shape,(16,1))
        key=torch.from_numpy(result['k_star']);module=model.transformer.h[17].mlp.c_proj
        with torch.no_grad():native=module(key)
        np.testing.assert_allclose(native.numpy(),result['wk_plus_bias'],atol=1e-6)
        expected=key@module.weight+module.bias
        torch.testing.assert_close(native,expected)
        np.testing.assert_allclose(result['wk_plus_bias']+result['U0']@(result['V0'].T@result['k_star']),result['v_star'],atol=2e-5,rtol=2e-5)
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'edit';meta=save(result,path)
            from edit_cert import backends
            loaded=backends.load(path)
            self.assertEqual(meta['content_sha256'],loaded.content_sha256())
            with torch.no_grad(),loaded.apply(SimpleNamespace(model=model)):
                after=module(key)
            np.testing.assert_allclose(after.numpy(),result['v_star'],atol=2e-5,rtol=2e-5)

    def test_subject_position_precedes_essence_end_and_targets_align(self):
        e=example(Tokenizer(),'X is a','X');self.assertEqual(e.position,0);self.assertLess(e.position,len(e.ids)-1)
        r=example(Tokenizer(),'X is','X',' Y');self.assertEqual(len(r.target_ids),2)
        self.assertEqual(r.ids,Tokenizer().encode('X is '))

    def test_clean_constraint_handles_negative_nonzero_dot(self):
        clean=np.array([1.,0.],dtype=np.float32);direction=np.array([-1.,0.],dtype=np.float32)
        base=np.array([2.,3.],dtype=np.float32);target=np.array([4.,7.],dtype=np.float32)
        u=affine_output_factor(target,base,clean,direction)
        np.testing.assert_array_equal(base+u*np.dot(direction,clean),target)
        with self.assertRaises(ValueError):affine_output_factor(target,base,clean,np.array([0.,1.],dtype=np.float32))


if __name__=='__main__':unittest.main()
