from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_p1_a3_spec_codex import rounding_example, trace_totals


class SpecAudit(unittest.TestCase):
    def test_archived_kl_and_decay_are_already_weighted(self):
        history = [dict(nll=.03, kl=.04, decay=.10)]
        self.assertAlmostEqual(trace_totals(history)[0], .17)
        self.assertNotAlmostEqual(trace_totals(history)[0], .03 + .0625 * .04 + .10)

    def test_cast_before_add_is_not_identical_to_mixed_add(self):
        result = rounding_example()
        self.assertFalse(result['equal_exactly'])
        self.assertEqual(result['cast_delta_before_add'], 1.)
        self.assertEqual(result['mixed_dtype_inplace_add'], 1.0009765625)


if __name__ == '__main__':
    unittest.main()
