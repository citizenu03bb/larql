"""Model-free boundary and tie tests for independent D1 statistics."""
import importlib.util
from pathlib import Path
import unittest

module = importlib.util.spec_from_file_location("d1", Path(__file__).parents[1] / "xcheck_d1.py")
d1 = importlib.util.module_from_spec(module)
module.loader.exec_module(d1)


class StatisticsTests(unittest.TestCase):
    def test_auc_accounts_for_ties_and_requires_both_labels(self):
        self.assertAlmostEqual(d1.auc([1, 2, 3, 0, 1, 2], [1, 1, 1, 0, 0, 0]), 7 / 9)
        self.assertIsNone(d1.auc([1, 2, 3, 0, 1], [1, 1, 1, 0, 0]))

    def test_spearman_uses_midrank_ties_and_rejects_constant_input(self):
        self.assertAlmostEqual(d1.spearman([1, 1, 3, 4], [2, 2, 6, 8]), 1.)
        self.assertAlmostEqual(d1.spearman([1, 1, 3, 4], [-2, -2, -6, -8]), -1.)
        self.assertIsNone(d1.spearman([1, 1, 1], [1, 2, 3]))

    def test_preregistered_thresholds_and_order_requirement(self):
        both = {o: 1 for o in d1.ORDERS}
        self.assertEqual(d1.classify(.499, 10, both, "PASS"), "K")
        self.assertEqual(d1.classify(.5, 1.999, both, "FAIL"), "V")
        self.assertEqual(d1.classify(.5, 2, both, "FAIL"), "S")
        self.assertEqual(d1.classify(.5, 2, both, "PASS"), "P")
        self.assertEqual(d1.classify(.5, 2, {d1.ORDERS[0]: 4}, "FAIL"), "U")


if __name__ == "__main__":
    unittest.main()
