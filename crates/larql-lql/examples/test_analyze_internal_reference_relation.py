#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_internal_reference_relation import (
    ENTITY_ALIAS_IDS,
    analyze,
    capture_key,
    first_alias_span,
    last_residual,
    reference_vector,
    verify_last_position_parity,
)


class InternalReferenceRelationTests(unittest.TestCase):
    def capture(self):
        positions = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        return {
            "prompt_ids": [2, *ENTITY_ALIAS_IDS[0], 99],
            "position_residuals": [[13, positions.tolist()]],
        }

    def test_finds_first_frozen_entity_alias(self):
        self.assertEqual(first_alias_span(self.capture()["prompt_ids"]), (1, 4))

    def test_previous_reference_is_penultimate_position(self):
        np.testing.assert_array_equal(
            reference_vector(self.capture(), 13, "previous"), [5.0, 6.0]
        )

    def test_mean_body_excludes_bos_and_last(self):
        np.testing.assert_array_equal(
            reference_vector(self.capture(), 13, "mean_body"), [3.0, 4.0]
        )

    def test_first_body_reference_is_position_one(self):
        np.testing.assert_array_equal(
            reference_vector(self.capture(), 13, "first_body"), [1.0, 2.0]
        )

    def test_entity_span_reference_uses_exact_alias_positions(self):
        np.testing.assert_array_equal(
            reference_vector(self.capture(), 13, "entity_span"), [3.0, 4.0]
        )

    def test_replay_last_position_must_exactly_match_baseline(self):
        capture = self.capture()
        capture.update(
            {
                "fold": 0,
                "cell": "11",
                "name": "canonical",
                "prompt": "prompt",
            }
        )
        baseline = {
            key: value for key, value in capture.items() if key != "position_residuals"
        }
        baseline["residuals"] = [[13, [7.0, 8.0]]]
        self.assertEqual(capture_key(capture), capture_key(baseline))
        np.testing.assert_array_equal(last_residual(baseline, 13), [7.0, 8.0])
        self.assertEqual(
            verify_last_position_parity([capture], [baseline], 13)["status"],
            "exact",
        )

    def test_replay_last_position_mismatch_is_infrastructure_inconclusive(self):
        capture = self.capture()
        capture.update(
            {
                "fold": 0,
                "cell": "11",
                "name": "canonical",
                "prompt": "prompt",
            }
        )
        baseline = {
            key: value for key, value in capture.items() if key != "position_residuals"
        }
        baseline["residuals"] = [[13, [7.0, 8.5]]]
        with self.assertRaisesRegex(ValueError, "infrastructure-inconclusive"):
            verify_last_position_parity([capture], [baseline], 13)

    def test_analysis_selects_first_strictly_separating_reference(self):
        captures = []
        baseline = []
        for fold in range(6):
            offset = float(fold * 3)
            for cell in ("11", "10", "01", "00"):
                relation = 1.0 if cell[1] == "1" else -1.0
                positions = [
                    [0.0, offset],
                    [0.0, offset],
                    [0.0, offset],
                    [0.0, offset],
                    [relation, offset],
                ]
                row = {
                    "fold": fold,
                    "cell": cell,
                    "name": f"fold-{fold}-{cell}",
                    "prompt": f"prompt-{fold}-{cell}",
                    "prompt_ids": [2, *ENTITY_ALIAS_IDS[0], 99],
                    "position_residuals": [[13, positions]],
                }
                captures.append(row)
                baseline.append(
                    {
                        **{k: v for k, v in row.items() if k != "position_residuals"},
                        "residuals": [[13, positions[-1]]],
                    }
                )

        report = analyze(captures, baseline)
        self.assertTrue(report["b_prime_survives"])
        self.assertEqual(report["selected_reference"], "previous")
        self.assertGreater(report["references"][0]["global_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
