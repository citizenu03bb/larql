#!/usr/bin/env python3

import copy
import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_counterfactual_entity_swap_interaction import analyze


class CounterfactualEntitySwapInteractionTests(unittest.TestCase):
    def synthetic(self, clean_layers=(11,)):
        clean_layers = set(clean_layers)
        captures = []
        for fold in range(6):
            for cell in ("11", "10", "01", "00"):
                entity = 1.0 if cell[0] == "1" else 0.0
                relation = 1.0 if cell[1] == "1" else 0.0
                prompt_ids = [2, 100 + fold, 88, 99]
                residuals = []
                position_residuals = []
                for layer in range(42):
                    interaction = 2.0 if layer in clean_layers else 0.1
                    entity_construction_nuisance = (
                        0.0 if layer in clean_layers else entity * float(fold)
                    )
                    final = [
                        entity_construction_nuisance + entity * relation * interaction,
                        entity * 0.5,
                        float(fold * 3),
                    ]
                    positions = [[0.0, 0.0, 0.0]] * 3 + [final]
                    residuals.append([layer, final])
                    position_residuals.append([layer, positions])
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

    def replace_layer_final(self, captures, layer, value_for):
        for capture in captures:
            value = value_for(capture)
            capture["residuals"][layer][1] = value
            capture["position_residuals"][layer][1][-1] = value

    def test_nominates_clean_interaction_with_adjacent_direction_continuity(self):
        report = analyze(self.synthetic(clean_layers=(11,)))
        self.assertEqual(report["configuration_count"], 42)
        self.assertEqual(report["positive_gap_layers"], [11])
        self.assertEqual(report["separation_passing_layers"], [11])
        self.assertEqual(report["passing_layers"], [11])
        self.assertEqual(report["nominated_layer"], 11)
        self.assertEqual(report["verdict"], "exploratory two-pass candidate nominated")
        layer = report["layers"][11]
        self.assertGreaterEqual(layer["margin_ratio"], 0.5)
        self.assertGreaterEqual(layer["best_adjacent_minimum_cosine"], 0.7)
        self.assertEqual(len(layer["rows"]), 6)
        self.assertEqual(len(report["adjacent_pairs"]), 41)

    def test_multiple_survivors_select_highest_ratio_then_lower_layer(self):
        report = analyze(self.synthetic(clean_layers=(11, 12)))
        self.assertEqual(report["passing_layers"], [11, 12])
        self.assertEqual(report["nominated_layer"], 11)

    def test_positive_gap_below_ratio_floor_is_not_nominated(self):
        captures = self.synthetic(clean_layers=(11,))

        def marginal(capture):
            entity = 1.0 if capture["cell"][0] == "1" else 0.0
            relation = 1.0 if capture["cell"][1] == "1" else 0.0
            return [entity * (relation * 2.0 + capture["fold"] * 0.3), entity * 0.5, 0.0]

        self.replace_layer_final(captures, 11, marginal)
        report = analyze(captures)
        layer = report["layers"][11]
        self.assertGreater(layer["global_gap"], 0.0)
        self.assertLess(layer["margin_ratio"], 0.5)
        self.assertFalse(layer["separation_passes"])
        self.assertNotIn(11, report["passing_layers"])

    def test_direction_discontinuity_blocks_nomination(self):
        captures = self.synthetic(clean_layers=(11,))

        def orthogonal(capture):
            entity = 1.0 if capture["cell"][0] == "1" else 0.0
            relation = 1.0 if capture["cell"][1] == "1" else 0.0
            return [entity * float(capture["fold"]), entity * relation * 0.1, 0.0]

        self.replace_layer_final(captures, 10, orthogonal)
        self.replace_layer_final(captures, 12, orthogonal)
        report = analyze(captures)
        layer = report["layers"][11]
        self.assertTrue(layer["separation_passes"])
        self.assertLess(layer["best_adjacent_minimum_cosine"], 0.7)
        self.assertFalse(layer["continuity_passes"])
        self.assertNotIn(11, report["passing_layers"])

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
