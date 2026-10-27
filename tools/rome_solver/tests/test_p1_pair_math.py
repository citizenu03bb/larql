import importlib.util
from pathlib import Path
import unittest
import numpy as np

module=importlib.util.spec_from_file_location('pairmath',Path(__file__).parents[1]/'p1_pair_math.py');m=importlib.util.module_from_spec(module);module.loader.exec_module(m)


class PairMath(unittest.TestCase):
    def test_reciprocal_factor_scaling_preserves_update_norm(self):
        u=np.array([[1.],[2.]]);v=np.array([[3.],[4.]])
        result=m.compare_factors(u,v,2*u,v/2)
        self.assertTrue(result['agrees']);self.assertAlmostEqual(result['norm_a'],result['norm_b'])

    def test_signs_are_not_flipped_and_ambiguous_ten_percent_is_reported(self):
        u=np.array([[1.]]);v=np.array([[1.]])
        self.assertFalse(m.compare_factors(u,v,-u,-v)['agrees'])
        self.assertIsNone(m.compare_factors(u,v,u*1.105,v)['agrees'])
        self.assertFalse(m.compare_factors(u,v,u*1.2,v)['agrees'])

    def test_affine_base_includes_bias(self):
        u=np.array([[2.],[3.]]);v=np.array([[1.],[0.]])
        base=np.array([7.,8.]);key=np.array([2.,4.]);target=base+np.array([4.,6.])
        self.assertTrue(m.affine_constraint(u,v,key,target,base)['passed'])
        self.assertFalse(m.affine_constraint(u,v,key,target,base-1)['passed'])


if __name__=='__main__':unittest.main()
