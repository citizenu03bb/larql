#!/usr/bin/env python3

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


EXAMPLES = Path(__file__).resolve().parent
REPOSITORY = EXAMPLES.parents[2]
sys.path.insert(0, str(EXAMPLES))

from analyze_factorized_relation import (
    compile_calibrated_detector,
    partition_suites,
    primary_score,
    select_thresholds,
    ternary_state,
)
from analyze_contextual_positions import matrix


class FactorizedRelationAnalysisTests(unittest.TestCase):
    def test_separation_gap_builds_an_abstention_zone(self):
        scores = np.asarray([-2.0, -1.0, 1.5, 3.0])
        labels = np.asarray([0, 0, 1, 1])

        no_fire, fire = select_thresholds(scores, labels, "separation_gap")

        self.assertEqual(no_fire, -1.0)
        self.assertEqual(fire, 1.5)
        self.assertEqual(ternary_state(-2.0, no_fire, fire), "no_fire")
        self.assertEqual(ternary_state(0.0, no_fire, fire), "abstain")
        self.assertEqual(ternary_state(2.0, no_fire, fire), "fire")

    def test_separation_gap_rejects_overlapping_training_scores(self):
        scores = np.asarray([-1.0, 0.5, 0.25, 2.0])
        labels = np.asarray([0, 0, 1, 1])

        with self.assertRaisesRegex(ValueError, "no strict separation gap"):
            select_thresholds(scores, labels, "separation_gap")

    def test_zero_false_positive_places_fire_above_negative_ceiling(self):
        scores = np.asarray([-1.0, 0.5, 0.25, 2.0])
        labels = np.asarray([0, 0, 1, 1])

        no_fire, fire = select_thresholds(scores, labels, "zero_false_positive")

        self.assertEqual(no_fire, 0.5)
        self.assertGreater(fire, no_fire)
        self.assertEqual(ternary_state(0.5, no_fire, fire), "no_fire")

    def test_last_representation_accepts_single_layer_residual_capture(self):
        captures = [
            {
                "residuals": [[13, [1.25, -0.5]]],
                "position_residuals": [],
            }
        ]

        np.testing.assert_array_equal(
            matrix(captures, layer=13, representation="last"),
            np.asarray([[1.25, -0.5]]),
        )

    def test_capture_file_partitions_by_construction_fold(self):
        captures = {
            "development": [
                {"name": "a", "fold": 0},
                {"name": "b", "fold": 1},
                {"name": "c", "fold": 0},
            ]
        }

        grouped = partition_suites(captures, "fold")

        self.assertEqual(sorted(grouped), ["development:0", "development:1"])
        self.assertEqual(
            [capture["name"] for capture in grouped["development:0"]],
            ["a", "c"],
        )

    def test_primary_score_penalizes_control_fires_before_coverage(self):
        safe = {
            "invalid_folds": [],
            "aggregate": {
                "control_fires": 0,
                "control_n": 6,
                "positive_fire_coverage": 5 / 6,
            },
        }
        unsafe = {
            "invalid_folds": [],
            "aggregate": {
                "control_fires": 1,
                "control_n": 6,
                "positive_fire_coverage": 1.0,
            },
        }

        self.assertEqual(primary_score(safe), 5 / 6)
        self.assertEqual(primary_score(unsafe), -1 / 6)
        self.assertGreater(primary_score(safe), primary_score(unsafe))

    def test_calibrated_compiler_emits_one_ternary_detector(self):
        def captures(values):
            return [
                {
                    "name": cell,
                    "prompt": f"prompt-{cell}",
                    "cell": cell,
                    "residuals": [[13, [value, 0.0]]],
                    "position_residuals": [],
                }
                for cell, value in values
            ]

        development = captures(
            [("11", 2.0), ("10", -2.0), ("01", 1.5), ("00", -1.5)]
        )
        calibration = captures(
            [("11", 1.5), ("10", -1.0), ("01", 1.0), ("00", -1.5)]
        )

        report = compile_calibrated_detector(
            development,
            calibration,
            layer=13,
            representation="last",
            method="raw_conjunction",
            minimum_normalized_gap=0.10,
        )

        self.assertTrue(report["passed"], report["reasons"])
        self.assertIsNotNone(report["detector"])
        self.assertEqual(
            {row["state"] for row in report["rows"] if row["cell"] == "11"},
            {"fire"},
        )

    def test_preregistered_material_is_checksum_bound_and_factorial(self):
        material = REPOSITORY / "docs/research-data/factorized-relation-v1"
        expected = {
            "development": (
                6,
                "ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243",
            ),
            "calibration": (
                2,
                "0a3ab50667b4d9612c9654437fadab54e9ac4754b491540af9670f08b351a1df",
            ),
            "holdout": (
                6,
                "8f5f774c6a7a296ba11408044fbcd8bdba99d365402a581e5011f68bebd8d5ed",
            ),
        }
        all_prompts = set()
        for name, (n_folds, expected_sha256) in expected.items():
            raw = (material / f"{name}.json").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha256)
            suite = json.loads(raw)
            self.assertEqual(len(suite["facts"]), 1)
            fact = suite["facts"][0]
            self.assertEqual(fact["expected_tokens"], 4)
            positives = [
                f"The {fact['relation']} of {fact['entity']} is",
                *fact["paraphrases"],
            ]
            self.assertEqual(len(positives), n_folds)
            self.assertEqual(len(suite["controls"]), n_folds * 3)
            for fold in range(n_folds):
                cells = sorted(
                    control["cell"]
                    for control in suite["controls"]
                    if control["fold"] == fold
                )
                self.assertEqual(cells, ["00", "01", "10"])
            prompts = positives + [
                control["prompt"] for control in suite["controls"]
            ]
            self.assertEqual(len(prompts), len(set(prompts)))
            self.assertTrue(all_prompts.isdisjoint(prompts))
            all_prompts.update(prompts)


if __name__ == "__main__":
    unittest.main()
