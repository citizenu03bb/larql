#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_differential_first_body_direction import (
    analyze,
    verify_matched_first_tokens,
)


class DifferentialFirstBodyDirectionTests(unittest.TestCase):
    def synthetic(self, mismatch=False):
        captures = []
        baseline = []
        for fold in range(6):
            construction_offset = float(fold * 5)
            for cell in ("11", "10", "01", "00"):
                relation = 1.0 if cell[1] == "1" else -1.0
                first_token = 100 + fold + (10 if cell[0] == "0" else 0)
                if mismatch and fold == 0 and cell == "10":
                    first_token += 1
                prompt_ids = [2, first_token, 88, 99]
                positions = [
                    [0.0, 0.0],
                    [0.0, construction_offset],
                    [0.0, construction_offset],
                    [relation, construction_offset],
                ]
                row = {
                    "fold": fold,
                    "cell": cell,
                    "name": f"fold-{fold}-{cell}",
                    "prompt": f"prompt-{fold}-{cell}",
                    "prompt_ids": prompt_ids,
                    "position_residuals": [[13, positions]],
                }
                captures.append(row)
                baseline.append(
                    {
                        **{k: v for k, v in row.items() if k != "position_residuals"},
                        "residuals": [[13, positions[-1]]],
                    }
                )
        return captures, baseline

    def test_matched_first_tokens_are_required(self):
        captures, _ = self.synthetic(mismatch=True)
        with self.assertRaisesRegex(ValueError, "different first body tokens"):
            verify_matched_first_tokens(captures)

    def test_differential_fit_separates_synthetic_signal(self):
        captures, baseline = self.synthetic()
        report = analyze(captures, baseline)
        self.assertTrue(report["passes"])
        self.assertGreater(report["global_gap"], 0.0)
        self.assertGreater(report["minimum_paired_delta"], 0.0)
        self.assertEqual(report["matched_first_body_tokens"]["matched_pairs"], 12)


if __name__ == "__main__":
    unittest.main()
