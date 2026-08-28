#!/usr/bin/env python3
"""Evaluate the frozen L29/L31/L33 late-decision relation atlas."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_factorized_relation import fit_relation, relation_scores, serializable
from analyze_contextual_positions import positions_at
from analyze_internal_reference_relation import validate_factorial_captures


CAPTURE_LAYERS = (29, 31, 33)
NOMINATION_PRIORITY = (31, 29, 33)


def validate_capture_shape(captures: list[dict]) -> list[int]:
    folds = validate_factorial_captures(captures)
    expected_layers = set(CAPTURE_LAYERS)
    dimensions = {}
    for capture in captures:
        name = capture.get("name", "<unnamed>")
        residual_layers = [int(item[0]) for item in capture.get("residuals", [])]
        position_layers = [
            int(item[0]) for item in capture.get("position_residuals", [])
        ]
        if len(residual_layers) != len(set(residual_layers)):
            raise ValueError(f"capture {name!r} has duplicate residual layers")
        if len(position_layers) != len(set(position_layers)):
            raise ValueError(f"capture {name!r} has duplicate position layers")
        if set(residual_layers) != expected_layers:
            raise ValueError(
                f"capture {name!r} residual layers differ from frozen atlas: "
                f"{sorted(residual_layers)}"
            )
        if set(position_layers) != expected_layers:
            raise ValueError(
                f"capture {name!r} position layers differ from frozen atlas: "
                f"{sorted(position_layers)}"
            )
        prompt_ids = capture.get("prompt_ids", [])
        if not prompt_ids:
            raise ValueError(f"capture {name!r} has no prompt IDs")
        for layer in CAPTURE_LAYERS:
            positions = positions_at(capture, layer)
            if len(positions) != len(prompt_ids):
                raise ValueError(
                    f"capture {name!r} L{layer} position count differs from "
                    f"prompt token count: {len(positions)} != {len(prompt_ids)}"
                )
            if positions.ndim != 2 or positions.shape[1] == 0:
                raise ValueError(f"capture {name!r} L{layer} has invalid geometry")
            if not np.all(np.isfinite(positions)):
                raise ValueError(f"capture {name!r} L{layer} contains non-finite values")
            dimensions.setdefault(layer, positions.shape[1])
            if dimensions[layer] != positions.shape[1]:
                raise ValueError(f"capture {name!r} L{layer} width changed")
    return folds


def paired_score_deltas(rows: list[dict]) -> list[dict]:
    deltas = []
    for fold in range(6):
        for entity_bit in ("1", "0"):
            present = next(
                row
                for row in rows
                if row["fold"] == fold and row["cell"] == entity_bit + "1"
            )
            absent = next(
                row
                for row in rows
                if row["fold"] == fold and row["cell"] == entity_bit + "0"
            )
            deltas.append(
                {
                    "fold": fold,
                    "entity_bit": entity_bit,
                    "delta": present["score"] - absent["score"],
                }
            )
    return deltas


def direction_cosines(directions: list[np.ndarray]) -> list[dict]:
    rows = []
    for left, right in itertools.combinations(range(len(directions)), 2):
        rows.append(
            {
                "left_fold": left,
                "right_fold": right,
                "cosine": float(directions[left] @ directions[right]),
            }
        )
    return rows


def analyze_layer(
    captures: list[dict], folds: list[int], layer: int
) -> dict:
    cells = np.asarray([capture["cell"] for capture in captures])
    capture_folds = np.asarray([int(capture["fold"]) for capture in captures])
    capture_array = np.asarray(captures, dtype=object)
    last = np.asarray(
        [positions_at(capture, layer)[-1] for capture in captures],
        dtype=np.float64,
    )

    rows = []
    directions = []
    for heldout in folds:
        train = capture_folds != heldout
        test = ~train
        detector = fit_relation(last[train], cells[train], "raw_conjunction")
        direction = detector["direction"]
        if not np.all(np.isfinite(direction)):
            raise ValueError(f"analysis-inconclusive: L{layer} direction is non-finite")
        if float(np.linalg.norm(direction)) == 0.0:
            raise ValueError(f"analysis-inconclusive: L{layer} direction is zero")
        directions.append(direction)
        scores = relation_scores(detector, last[test])
        for capture, score in zip(capture_array[test], scores):
            rows.append(
                {
                    "fold": heldout,
                    "name": capture["name"],
                    "prompt": capture["prompt"],
                    "cell": capture["cell"],
                    "score": float(score),
                }
            )

    relation_present = [row["score"] for row in rows if row["cell"][1] == "1"]
    relation_absent = [row["score"] for row in rows if row["cell"][1] == "0"]
    eligible_positive = [row["score"] for row in rows if row["cell"] == "11"]
    eligible_control = [row["score"] for row in rows if row["cell"] == "10"]
    paired = paired_score_deltas(rows)
    cosines = direction_cosines(directions)
    global_gap = min(relation_present) - max(relation_absent)
    eligible_gap = min(eligible_positive) - max(eligible_control)
    minimum_paired_delta = min(row["delta"] for row in paired)

    return {
        "layer": layer,
        "representation": "raw pre-FFN residual at final prompt position",
        "direction_method": "raw_conjunction fitted LOCO",
        "global_gap": global_gap,
        "eligible_gap": eligible_gap,
        "minimum_paired_delta": minimum_paired_delta,
        "passes": global_gap > 0.0 and minimum_paired_delta > 0.0,
        "paired_score_deltas": paired,
        "minimum_direction_cosine": min(row["cosine"] for row in cosines),
        "direction_cosines": cosines,
        "rows": rows,
    }


def analyze(captures: list[dict]) -> dict:
    folds = validate_capture_shape(captures)
    by_layer = {
        layer: analyze_layer(captures, folds, layer) for layer in CAPTURE_LAYERS
    }
    nominated_layer = next(
        (layer for layer in NOMINATION_PRIORITY if by_layer[layer]["passes"]),
        None,
    )
    return serializable(
        {
            "version": 1,
            "epistemic_status": (
                "exploratory nomination gate on spent construction groups; not "
                "calibration, prospective evidence, certification, or holdout evidence"
            ),
            "configuration_count": len(CAPTURE_LAYERS),
            "capture_layers": list(CAPTURE_LAYERS),
            "nomination_priority": list(NOMINATION_PRIORITY),
            "nominated_layer": nominated_layer,
            "verdict": (
                "exploratory candidate nominated"
                if nominated_layer is not None
                else "exploratory candidate rejected"
            ),
            "layers": [by_layer[layer] for layer in CAPTURE_LAYERS],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.captures.read_text()))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
