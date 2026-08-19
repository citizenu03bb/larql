#!/usr/bin/env python3
"""Score untouched residual captures with a frozen contextual detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detector", type=Path)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    detector = json.loads(args.detector.read_text())
    captures = json.loads(args.captures.read_text())
    layer = detector["layer"]
    center = np.asarray(detector["center"], dtype=np.float64)
    kind = detector.get("kind", "single")
    if kind == "single":
        direction = np.asarray(detector["direction"], dtype=np.float64)
        threshold = float(detector["threshold"])
    elif kind == "conjunction":
        entity_direction = np.asarray(
            detector["entity_direction"], dtype=np.float64
        )
        relation_direction = np.asarray(
            detector["relation_direction"], dtype=np.float64
        )
        entity_threshold = float(detector["entity_threshold"])
        relation_threshold = float(detector["relation_threshold"])
    else:
        raise ValueError(f"unsupported detector kind: {kind}")

    rows = []
    for capture in captures:
        residual_by_layer = {item[0]: item[1] for item in capture["residuals"]}
        residual = np.asarray(residual_by_layer[layer], dtype=np.float64)
        centered = residual - center
        if kind == "single":
            score = float(centered @ direction - threshold)
        else:
            score = float(
                min(
                    centered @ entity_direction - entity_threshold,
                    centered @ relation_direction - relation_threshold,
                )
            )
        rows.append(
            {
                "name": capture["name"],
                "cell": capture["cell"],
                "prompt": capture["prompt"],
                "score": score,
                "margin_to_threshold": score,
                "passes": score >= 0,
            }
        )

    positive = [row["score"] for row in rows if row["cell"] == "11"]
    negative = [row["score"] for row in rows if row["cell"] != "11"]
    report = {
        "layer": layer,
        "detector_kind": kind,
        "threshold_frozen_before_validation": (
            threshold if kind == "single" else 0.0
        ),
        "n_captures": len(rows),
        "all_classified_correctly": all(
            row["passes"] == (row["cell"] == "11") for row in rows
        ),
        "validation_margin": float(min(positive) - max(negative)),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
