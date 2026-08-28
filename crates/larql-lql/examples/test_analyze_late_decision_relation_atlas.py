#!/usr/bin/env python3

import copy
import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_late_decision_relation_atlas import analyze


class LateDecisionRelationAtlasTests(unittest.TestCase):
    def synthetic(self):
        captures = []
        for fold in range(6):
            construction_offset = float(fold * 3)
            for cell in ("11", "10", "01", "00"):
                relation = 1.0 if cell[1] == "1" else -1.0
                entity = 0.25 if cell[0] == "1" else -0.25
                prompt_ids = [2, 100 + fold, 88, 99]
                position_residuals = []
                residuals = []
                for layer in (29, 31, 33):
                    scale = 1.0 + (layer - 29) * 0.1
                    positions = [
                        [0.0, 0.0, 0.0],
                        [0.0, construction_offset, entity],
                        [0.0, construction_offset, entity],
                        [relation * scale, construction_offset, entity],
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

    def test_all_layers_are_reported_and_l31_has_nomination_priority(self):
        report = analyze(self.synthetic())
        self.assertEqual(report["configuration_count"], 3)
        self.assertEqual(report["capture_layers"], [29, 31, 33])
        self.assertEqual(report["nomination_priority"], [31, 29, 33])
        self.assertEqual(report["nominated_layer"], 31)
        self.assertEqual(report["verdict"], "exploratory candidate nominated")
        for layer in report["layers"]:
            self.assertTrue(layer["passes"])
            self.assertGreater(layer["global_gap"], 0.0)
            self.assertGreater(layer["minimum_paired_delta"], 0.0)
            self.assertEqual(len(layer["rows"]), 24)
            self.assertEqual(len(layer["paired_score_deltas"]), 12)

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
