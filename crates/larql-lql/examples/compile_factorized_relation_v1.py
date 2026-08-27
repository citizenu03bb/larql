#!/usr/bin/env python3
"""Compile only the preregistered factorized-relation-v1 detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_factorized_relation import compile_calibrated_detector


LAYER = 13
REPRESENTATION = "last"
METHOD = "raw_conjunction"
MINIMUM_NORMALIZED_GAP = 0.10
BASE_BINDING = (
    "sha256-manifest:"
    "c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a"
)
ENTITY = "Archive Zephyr"
ENTITY_ALIAS_IDS = (
    (39140, 7733, 109999),
    (40187, 7733, 109999),
)
TARGET = "Nora Veldt"
TARGET_IDS = (107545, 782, 7337, 236745)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("development", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    development = json.loads(args.development.read_text())
    calibration = json.loads(args.calibration.read_text())
    report = compile_calibrated_detector(
        development,
        calibration,
        layer=LAYER,
        representation=REPRESENTATION,
        method=METHOD,
        minimum_normalized_gap=MINIMUM_NORMALIZED_GAP,
    )
    report["binding"] = {
        "base": BASE_BINDING,
        "entity": ENTITY,
        "entity_alias_ids": ENTITY_ALIAS_IDS,
        "target": TARGET,
        "target_ids": TARGET_IDS,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "reasons", "metrics")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
