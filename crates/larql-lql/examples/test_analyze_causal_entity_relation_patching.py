#!/usr/bin/env python3

import copy
import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_causal_entity_relation_patching import analyze


SITES = (11, 13, 12, 33)
ARMS = (
    "matched_forward",
    "matched_inverse",
    "positive_self",
    "control_self",
    "entity_absent_forward",
    "entity_absent_inverse",
    "nonpaired_forward",
    "nonpaired_inverse",
)


class CausalEntityRelationPatchingTests(unittest.TestCase):
    def synthetic(self):
        baselines = []
        for fold in range(6):
            for cell in ("11", "10", "01", "00"):
                digest = f"digest-{fold}-{cell}"
                baselines.append(
                    {
                        "fold": fold,
                        "cell": cell,
                        "target_logit": 1.0,
                        "distractor_logit": 0.0,
                        "gap": 1.0,
                        "full_logits_sha256": digest,
                        "noop_full_logits_sha256": digest,
                        "noop_digest_identical": True,
                    }
                )

        strengths = {11: 4.0, 13: 3.0, 12: 1.0, 33: 5.0}
        arms = []
        for site in SITES:
            for fold in range(6):
                for arm in ARMS:
                    if arm == "matched_forward":
                        delta_target = strengths[site]
                    elif arm == "matched_inverse":
                        delta_target = -strengths[site]
                    elif arm in ("positive_self", "control_self"):
                        delta_target = 0.0
                    else:
                        delta_target = 0.1
                    arms.append(
                        {
                            "site": site,
                            "fold": fold,
                            "arm": arm,
                            "recipient_position": 7,
                            "applied": [[site, 7]],
                            "rejected": [],
                            "target_logit": 1.0 + delta_target,
                            "distractor_logit": 0.0,
                            "gap": 1.0 + delta_target,
                            "delta_target": delta_target,
                            "delta_distractor": 0.0,
                            "delta_gap": delta_target,
                        }
                    )
        return {
            "schema": "larql.causal_entity_relation_patching.v1",
            "sites": list(SITES),
            "target_id": 107545,
            "distractor_id": 506,
            "baseline_forwards": 24,
            "noop_forwards": 24,
            "patched_forwards": 192,
            "total_forwards": 240,
            "baselines": baselines,
            "arms": arms,
        }

    def arm(self, run, site, fold, arm):
        return next(
            row
            for row in run["arms"]
            if row["site"] == site and row["fold"] == fold and row["arm"] == arm
        )

    def test_fixed_priority_selects_and_confirms_l11(self):
        report = analyze(self.synthetic())
        self.assertEqual(report["candidate_discovery_passes"], {"11": True, "13": True})
        self.assertEqual(report["selected_discovery_layer"], 11)
        self.assertTrue(report["selected_confirmation_passes"])
        self.assertEqual(report["nominated_layer"], 11)
        self.assertEqual(report["verdict"], "exploratory causal mediator nominated")

    def test_l13_cannot_replace_selected_l11_after_confirmation_failure(self):
        run = self.synthetic()
        row = self.arm(run, 11, 4, "matched_forward")
        row["target_logit"] = 0.0
        row["gap"] = 0.0
        row["delta_target"] = -1.0
        row["delta_gap"] = -1.0
        report = analyze(run)
        self.assertEqual(report["candidate_discovery_passes"], {"11": True, "13": True})
        self.assertEqual(report["selected_discovery_layer"], 11)
        self.assertFalse(report["selected_confirmation_passes"])
        self.assertIsNone(report["nominated_layer"])
        self.assertEqual(report["verdict"], "exploratory causal candidate rejected")

    def test_l13_is_selected_only_when_l11_fails_discovery(self):
        run = self.synthetic()
        row = self.arm(run, 11, 0, "matched_inverse")
        row["target_logit"] = 2.0
        row["gap"] = 2.0
        row["delta_target"] = 1.0
        row["delta_gap"] = 1.0
        report = analyze(run)
        self.assertEqual(report["candidate_discovery_passes"], {"11": False, "13": True})
        self.assertEqual(report["selected_discovery_layer"], 13)
        self.assertEqual(report["nominated_layer"], 13)

    def test_self_patch_drift_is_analysis_inconclusive(self):
        run = self.synthetic()
        row = self.arm(run, 13, 2, "positive_self")
        row["delta_gap"] = 0.00011
        with self.assertRaisesRegex(ValueError, "analysis-inconclusive.*self-patch drift"):
            analyze(run)

    def test_missing_arm_and_failed_noop_are_analysis_inconclusive(self):
        missing = self.synthetic()
        missing["arms"].pop()
        with self.assertRaisesRegex(ValueError, "analysis-inconclusive.*192 patched arms"):
            analyze(missing)

        failed_noop = self.synthetic()
        failed_noop["baselines"][0]["noop_digest_identical"] = False
        with self.assertRaisesRegex(ValueError, "analysis-inconclusive.*digest identity"):
            analyze(failed_noop)

    def test_non_finite_metric_is_analysis_inconclusive(self):
        run = self.synthetic()
        self.arm(run, 11, 0, "matched_forward")["delta_target"] = float("nan")
        with self.assertRaisesRegex(ValueError, "analysis-inconclusive.*non-finite"):
            analyze(run)

    def test_duplicate_arm_is_analysis_inconclusive(self):
        run = self.synthetic()
        run["arms"][-1] = copy.deepcopy(run["arms"][0])
        with self.assertRaisesRegex(ValueError, "analysis-inconclusive.*duplicate patched arm"):
            analyze(run)


if __name__ == "__main__":
    unittest.main()
