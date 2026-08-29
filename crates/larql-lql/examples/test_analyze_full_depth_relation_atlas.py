#!/usr/bin/env python3

import copy
import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_full_depth_relation_atlas import analyze


class FullDepthRelationAtlasTests(unittest.TestCase):
    def synthetic(self, passing_layers=range(42)):
        passing_layers = set(passing_layers)
        captures = []
        for fold in range(6):
            construction_offset = float(fold * 3)
            for cell in ("11", "10", "01", "00"):
                relation = 1.0 if cell[1] == "1" else -1.0
                entity = 0.25 if cell[0] == "1" else -0.25
                prompt_ids = [2, 100 + fold, 88, 99]
                position_residuals = []
                residuals = []
                for layer in range(42):
                    relation_value = relation if layer in passing_layers else relation * 0.1
                    nuisance = 0.0 if layer in passing_layers else construction_offset
                    positions = [
                        [0.0, 0.0, 0.0],
                        [nuisance, 0.0, entity],
                        [nuisance, 0.0, entity],
                        [relation_value + nuisance, 0.0, entity],
                    ]
                    position_residuals.append([layer, positions])
                    residuals.append([layer, positions[-1]])
                captures.append(
                    {
                        "fold": fold,
                        "cell": cell,
                        "name": f"fold-{fold}-{cell}",
                        "prompt": f"prompt-{fold}-{cell}",
                        "prompt_ids": prompt_ids,
                        "residuals": residuals,
                        "position_residuals": position_residuals,
                    }
                )
        return captures

    def test_reports_all_layers_and_nominates_middle_of_earliest_window(self):
        report = analyze(self.synthetic(range(20, 24)))
        self.assertEqual(report["configuration_count"], 42)
        self.assertEqual(report["capture_layers"], list(range(42)))
        self.assertEqual(report["passing_layers"], [20, 21, 22, 23])
        self.assertEqual(report["stable_runs"], [[20, 21, 22, 23]])
        self.assertEqual(report["nominated_layer"], 21)
        self.assertEqual(report["verdict"], "exploratory stable band nominated")
        self.assertEqual(len(report["layers"]), 42)

    def test_isolated_positive_is_diagnostic_only(self):
        report = analyze(self.synthetic([31]))
        self.assertEqual(report["passing_layers"], [31])
        self.assertEqual(report["passing_runs"], [[31]])
        self.assertEqual(report["stable_runs"], [])
        self.assertIsNone(report["nominated_layer"])
        self.assertEqual(
            report["verdict"], "exploratory map positive but no stable band"
        )

    def test_missing_or_duplicate_frozen_layer_is_rejected(self):
        missing = self.synthetic()
        missing[0]["position_residuals"] = missing[0]["position_residuals"][:-1]
        with self.assertRaisesRegex(ValueError, "position layers differ"):
            analyze(missing)

        duplicate = self.synthetic()
        duplicate[0]["position_residuals"].append(
            copy.deepcopy(duplicate[0]["position_residuals"][0])
        )
        with self.assertRaisesRegex(ValueError, "duplicate position layers"):
            analyze(duplicate)

    def test_non_finite_capture_is_rejected(self):
        captures = self.synthetic()
        captures[0]["position_residuals"][0][1][-1][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            analyze(captures)


if __name__ == "__main__":
    unittest.main()
