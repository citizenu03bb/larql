import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from review_p1_a3 import factors


class Decomposition(unittest.TestCase):
    def test_norm_ratio_splits_into_shift_and_key_factors(self):
        # Shifts have norms 10 and 2; clean/mean projections are 2 and 6.
        delta = np.array([6., 8.]); peer = np.array([0., 2.])
        result = factors(delta, peer, np.array([1., 0.]), np.array([2., 3.]), np.array([6., 9.]), 15.)
        self.assertEqual(result['objective_factor'], 5.)
        self.assertEqual(result['key_factor'], 3.)
        self.assertEqual(result['product_relative_error'], 0.)

    def test_norm_factor_uses_absolute_signed_projection(self):
        result = factors(np.array([2.]), np.array([1.]), np.array([1.]), np.array([-1.]), np.array([3.]), 6.)
        self.assertEqual(result['key_factor'], 3.)
        self.assertEqual(result['product_relative_error'], 0.)


if __name__ == '__main__':
    unittest.main()
