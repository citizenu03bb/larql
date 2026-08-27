#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from analyze_entity_span_reference_completion import (
    ENTITY_ALIASES,
    analyze,
    exact_entity_span,
)


ALIASES = {name: list(ids) for name, ids in ENTITY_ALIASES}


class EntitySpanReferenceCompletionTests(unittest.TestCase):
    def test_finds_each_frozen_alias(self):
        for name, alias in ALIASES.items():
            with self.subTest(name=name):
                self.assertEqual(
                    exact_entity_span([2, 77, *alias, 99]),
                    (name, 2, 2 + len(alias)),
                )

    def test_rejects_missing_alias(self):
        with self.assertRaisesRegex(ValueError, "no frozen Archive entity alias"):
            exact_entity_span([2, 55, 56, 99])

    def test_rejects_multiple_aliases(self):
        with self.assertRaisesRegex(ValueError, "exactly one entity alias"):
            exact_entity_span(
                [
                    2,
                    *ALIASES["archive_zephyr_start"],
                    *ALIASES["archive_nimbus_space"],
                    99,
                ]
            )

    def test_full_analysis_uses_both_entities_and_separates_synthetic_signal(self):
        captures = []
        baseline = []
        for fold in range(6):
            construction_offset = float(fold * 4)
            for cell in ("11", "10", "01", "00"):
                relation = 1.0 if cell[1] == "1" else -1.0
                alias_name = (
                    "archive_zephyr_start"
                    if cell[0] == "1"
                    else "archive_nimbus_start"
                )
                prompt_ids = [2, *ALIASES[alias_name], 99]
                positions = [
                    [0.0, construction_offset]
                    for _ in range(len(prompt_ids))
                ]
                positions[-1] = [relation, construction_offset]
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

        report = analyze(captures, baseline)
        self.assertTrue(report["passes"])
        self.assertGreater(report["global_gap"], 0.0)
        self.assertEqual(report["baseline_last_position_parity"]["status"], "exact")
        self.assertEqual(
            {row["entity_alias"] for row in report["rows"]},
            {"archive_zephyr_start", "archive_nimbus_start"},
        )


if __name__ == "__main__":
    unittest.main()
