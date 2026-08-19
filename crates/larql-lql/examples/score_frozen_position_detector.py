#!/usr/bin/env python3
"""Score untouched all-position captures with one frozen detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_contextual_positions import represent, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detector", type=Path)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    detector = json.loads(args.detector.read_text())
    captures = json.loads(args.captures.read_text())
    layer = int(detector["layer"])
    representation = detector["representation"]
    numeric = {
        key: np.asarray(value, dtype=np.float64)
        if key
        in {
            "center",
            "entity_direction",
            "relation_direction",
            "entity_present",
            "entity_absent",
            "relation_present",
            "relation_absent",
        }
        else value
        for key, value in detector.items()
    }
    if "prototypes" in numeric:
        numeric["prototypes"] = {
            cell: np.asarray(value, dtype=np.float64)
            for cell, value in numeric["prototypes"].items()
        }

    rows_matrix = []
    for capture in captures:
        by_layer = {item[0]: item[1] for item in capture["position_residuals"]}
        positions = np.asarray(by_layer[layer], dtype=np.float64)
        rows_matrix.append(represent(positions, representation))
    scores = score(numeric, np.asarray(rows_matrix, dtype=np.float64))

    rows = []
    for capture, value in zip(captures, scores):
        expected = capture["cell"] == "11"
        passed = bool(value >= 0)
        rows.append(
            {
                "name": capture["name"],
                "cell": capture["cell"],
                "prompt": capture["prompt"],
                "score": float(value),
                "passes": passed,
                "correct": passed == expected,
            }
        )
    positives = [row["score"] for row in rows if row["cell"] == "11"]
    negatives = [row["score"] for row in rows if row["cell"] != "11"]
    signed = [
        row["score"] if row["cell"] == "11" else -row["score"] for row in rows
    ]
    report = {
        "layer": layer,
        "representation": representation,
        "method": detector["method"],
        "n_captures": len(rows),
        "all_classified_correctly": all(row["correct"] for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "decision_margin": float(min(signed)),
        "ranking_margin": float(min(positives) - max(negatives)),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
