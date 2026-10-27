from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from xcheck_p1_5 import rates, paper_threshold


class P15Reading(unittest.TestCase):
    def test_zero_margin_is_neither_efficacy_nor_preservation(self):
        r = rates(dict(S1=0., E1=0., O1=0.))
        self.assertEqual([r[k]['successes'] for k in ('ES','PS','NS')], [0,0,0])
        self.assertFalse(paper_threshold(r))

    def test_missing_seen_item_cannot_satisfy_the_rule(self):
        r = rates(dict(E1=2., E2=3., O4=-1.))
        self.assertIsNone(r['ES']['rate'])
        self.assertIsNone(paper_threshold(r))

    def test_exact_seventy_five_percent_is_inclusive(self):
        r = rates(dict(S1=1., E1=1., E2=1., E3=1., E4=-1., O1=-1., O2=-1., O3=-1., O4=1.))
        self.assertTrue(paper_threshold(r))


if __name__ == '__main__':
    unittest.main()
