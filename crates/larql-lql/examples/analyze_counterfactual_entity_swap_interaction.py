#!/usr/bin/env python3
"""Evaluate the frozen counterfactual entity-swap interaction atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_contextual_positions import positions_at
from analyze_factorized_relation import serializable
from analyze_full_depth_relation_atlas import (
    CAPTURE_LAYERS,
    validate_capture_shape,
)


MARGIN_RATIO_FLOOR = 0.5
ADJACENT_COSINE_FLOOR = 0.7


def unit(vector: np.ndarray, *, context: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"analysis-inconclusive: {context} direction is zero or non-finite")
    return vector / norm


def cell_vectors(captures: list[dict], layer: int) -> dict[int, dict[str, np.ndarray]]:
    by_fold: dict[int, dict[str, np.ndarray]] = {}
    for capture in captures:
        fold = int(capture["fold"])
        cell = capture["cell"]
        by_fold.setdefault(fold, {})[cell] = np.asarray(
            positions_at(capture, layer)[-1], dtype=np.float64
        )
    return by_fold


def analyze_layer(captures: list[dict], folds: list[int], layer: int) -> dict:
    cells = cell_vectors(captures, layer)
    swaps = {
        fold: {
            "present": cells[fold]["11"] - cells[fold]["01"],
            "absent": cells[fold]["10"] - cells[fold]["00"],
        }
        for fold in folds
    }

    rows = []
    directions = []
    for heldout in folds:
        training = [fold for fold in folds if fold != heldout]
        present_train = np.asarray([swaps[fold]["present"] for fold in training])
        absent_train = np.asarray([swaps[fold]["absent"] for fold in training])
        center = np.mean(np.concatenate([present_train, absent_train], axis=0), axis=0)
        direction = unit(
            np.mean(present_train - center, axis=0)
            - np.mean(absent_train - center, axis=0),
            context=f"L{layer} fold {heldout}",
        )
        directions.append(direction)
        present_score = float((swaps[heldout]["present"] - center) @ direction)
        absent_score = float((swaps[heldout]["absent"] - center) @ direction)
        if not np.isfinite(present_score) or not np.isfinite(absent_score):
            raise ValueError(f"analysis-inconclusive: L{layer} fold {heldout} score is non-finite")
        rows.append(
            {
                "fold": heldout,
                "present_score": present_score,
                "absent_score": absent_score,
                "paired_delta": present_score - absent_score,
            }
        )

    present_scores = [row["present_score"] for row in rows]
    absent_scores = [row["absent_score"] for row in rows]
    paired_deltas = [row["paired_delta"] for row in rows]
    global_gap = min(present_scores) - max(absent_scores)
    minimum_paired_delta = min(paired_deltas)
    margin_ratio = (
        global_gap / minimum_paired_delta if minimum_paired_delta > 0.0 else None
    )
    separation_passes = (
        global_gap > 0.0
        and minimum_paired_delta > 0.0
        and margin_ratio is not None
        and margin_ratio >= MARGIN_RATIO_FLOOR
    )

    return {
        "layer": layer,
        "representation": (
            "raw final-position entity-swap difference with LOCO centering"
        ),
        "direction_method": "LOCO entity-by-relation interaction",
        "global_gap": global_gap,
        "minimum_paired_delta": minimum_paired_delta,
        "margin_ratio": margin_ratio,
        "separation_passes": separation_passes,
        "rows": rows,
        "_directions": directions,
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    value = float(left @ right)
    if not np.isfinite(value):
        raise ValueError("analysis-inconclusive: adjacent direction cosine is non-finite")
    return value


def add_continuity(layer_reports: list[dict]) -> list[dict]:
    by_layer = {report["layer"]: report for report in layer_reports}
    adjacent_pairs = []
    for left in CAPTURE_LAYERS[:-1]:
        right = left + 1
        fold_cosines = [
            cosine(a, b)
            for a, b in zip(
                by_layer[left]["_directions"], by_layer[right]["_directions"]
            )
        ]
        adjacent_pairs.append(
            {
                "left_layer": left,
                "right_layer": right,
                "minimum_fold_cosine": min(fold_cosines),
                "fold_cosines": fold_cosines,
            }
        )

    pair_lookup = {
        (pair["left_layer"], pair["right_layer"]): pair for pair in adjacent_pairs
    }
    for report in layer_reports:
        layer = report["layer"]
        neighbors = []
        if layer > CAPTURE_LAYERS[0]:
            pair = pair_lookup[(layer - 1, layer)]
            neighbors.append(
                {
                    "layer": layer - 1,
                    "minimum_fold_cosine": pair["minimum_fold_cosine"],
                }
            )
        if layer < CAPTURE_LAYERS[-1]:
            pair = pair_lookup[(layer, layer + 1)]
            neighbors.append(
                {
                    "layer": layer + 1,
                    "minimum_fold_cosine": pair["minimum_fold_cosine"],
                }
            )
        best = max(
            neighbors,
            key=lambda item: (item["minimum_fold_cosine"], -item["layer"]),
        )
        report["adjacent_layers"] = neighbors
        report["best_adjacent_layer"] = best["layer"]
        report["best_adjacent_minimum_cosine"] = best["minimum_fold_cosine"]
        report["continuity_passes"] = (
            best["minimum_fold_cosine"] >= ADJACENT_COSINE_FLOOR
        )
        report["passes"] = report["separation_passes"] and report["continuity_passes"]
        del report["_directions"]
    return adjacent_pairs


def analyze(captures: list[dict]) -> dict:
    folds = validate_capture_shape(captures)
    layer_reports = [analyze_layer(captures, folds, layer) for layer in CAPTURE_LAYERS]
    adjacent_pairs = add_continuity(layer_reports)
    passing = [report for report in layer_reports if report["passes"]]
    nominated = (
        max(passing, key=lambda report: (report["margin_ratio"], -report["layer"]))
        if passing
        else None
    )

    return serializable(
        {
            "version": 1,
            "epistemic_status": (
                "exploratory counterfactual entity-swap interaction on spent groups; "
                "not calibration, prospective evidence, certification, or holdout evidence"
            ),
            "configuration_count": len(CAPTURE_LAYERS),
            "capture_layers": list(CAPTURE_LAYERS),
            "target_entity": "Archive Zephyr",
            "reference_entity": "Archive Nimbus",
            "margin_ratio_floor": MARGIN_RATIO_FLOOR,
            "adjacent_cosine_floor": ADJACENT_COSINE_FLOOR,
            "positive_gap_layers": [
                report["layer"] for report in layer_reports if report["global_gap"] > 0.0
            ],
            "separation_passing_layers": [
                report["layer"] for report in layer_reports if report["separation_passes"]
            ],
            "passing_layers": [report["layer"] for report in passing],
            "nominated_layer": nominated["layer"] if nominated else None,
            "verdict": (
                "exploratory two-pass candidate nominated"
                if nominated
                else "exploratory candidate rejected"
            ),
            "adjacent_pairs": adjacent_pairs,
            "layers": layer_reports,
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
