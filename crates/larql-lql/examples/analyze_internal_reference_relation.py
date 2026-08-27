#!/usr/bin/env python3
"""Evaluate the frozen B-prime internal-reference relation hypotheses.

This is a post-closure architecture diagnostic on spent prompts.  It fits the
relation direction on the last-position residuals of the training construction
groups, then evaluates a fixed list of within-prompt reference vectors on the
held-out group.  It never fits a threshold: survival requires a strictly
positive global separation gap across all held-out relation-present and
relation-absent cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_factorized_relation import fit_relation, serializable
from analyze_contextual_positions import positions_at


REFERENCES = (
    "previous",
    "mean_body",
    "first_body",
    "entity_span",
)
ENTITY_ALIAS_IDS = (
    (39140, 7733, 109999),
    (40187, 7733, 109999),
)


def first_alias_span(prompt_ids: list[int]) -> tuple[int, int]:
    for start in range(len(prompt_ids)):
        for alias in ENTITY_ALIAS_IDS:
            end = start + len(alias)
            if tuple(prompt_ids[start:end]) == alias:
                return start, end
    raise ValueError("prompt contains no frozen Archive Zephyr entity alias")


def reference_vector(capture: dict, layer: int, reference: str) -> np.ndarray:
    positions = positions_at(capture, layer)
    if len(positions) < 3:
        raise ValueError("internal-reference analysis requires at least 3 positions")
    if reference == "previous":
        return positions[-2]
    if reference == "mean_body":
        return np.mean(positions[1:-1], axis=0)
    if reference == "first_body":
        return positions[1]
    if reference == "entity_span":
        start, end = first_alias_span(capture["prompt_ids"])
        return np.mean(positions[start:end], axis=0)
    raise ValueError(f"unknown internal reference: {reference}")


def capture_key(capture: dict) -> tuple[int, str, str, str]:
    return (
        int(capture["fold"]),
        capture["cell"],
        capture["name"],
        capture["prompt"],
    )


def last_residual(capture: dict, layer: int) -> np.ndarray:
    residuals = {int(item[0]): item[1] for item in capture["residuals"]}
    if layer not in residuals:
        raise ValueError(f"baseline capture has no layer {layer} residual")
    return np.asarray(residuals[layer], dtype=np.float64)


def verify_last_position_parity(
    captures: list[dict], baseline: list[dict], layer: int
) -> dict:
    baseline_by_key = {capture_key(capture): capture for capture in baseline}
    capture_keys = {capture_key(capture) for capture in captures}
    if capture_keys != set(baseline_by_key):
        missing = sorted(set(baseline_by_key) - capture_keys)
        unexpected = sorted(capture_keys - set(baseline_by_key))
        raise ValueError(
            "replay/baseline capture identities differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    mismatches = []
    for capture in captures:
        key = capture_key(capture)
        replay_last = positions_at(capture, layer)[-1]
        baseline_last = last_residual(baseline_by_key[key], layer)
        if not np.array_equal(replay_last, baseline_last):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "infrastructure-inconclusive: replay last-position residuals differ "
            f"from the frozen baseline for {mismatches}"
        )
    return {
        "status": "exact",
        "layer": layer,
        "matched_captures": len(captures),
    }


def validate_factorial_captures(captures: list[dict]) -> list[int]:
    folds = sorted({int(capture["fold"]) for capture in captures})
    if folds != list(range(6)):
        raise ValueError(f"expected frozen folds 0..5, got {folds}")
    expected_cells = {"11", "10", "01", "00"}
    for fold in folds:
        fold_cells = {
            capture["cell"] for capture in captures if int(capture["fold"]) == fold
        }
        if fold_cells != expected_cells:
            raise ValueError(
                f"fold {fold} must contain the four factorial cells, got "
                f"{sorted(fold_cells)}"
            )
    if len(captures) != 24:
        raise ValueError(f"expected exactly 24 frozen captures, got {len(captures)}")
    return folds


def analyze(
    captures: list[dict], baseline: list[dict], layer: int = 13
) -> dict:
    folds = validate_factorial_captures(captures)
    parity = verify_last_position_parity(captures, baseline, layer)
    cells = np.asarray([capture["cell"] for capture in captures])
    capture_folds = np.asarray([int(capture["fold"]) for capture in captures])
    last = np.asarray(
        [positions_at(capture, layer)[-1] for capture in captures],
        dtype=np.float64,
    )

    reports = []
    for reference in REFERENCES:
        rows = []
        for heldout in folds:
            train = capture_folds != heldout
            test = ~train
            detector = fit_relation(last[train], cells[train], "raw_conjunction")
            heldout_last = last[test]
            heldout_reference = np.asarray(
                [
                    reference_vector(capture, layer, reference)
                    for capture in np.asarray(captures, dtype=object)[test]
                ],
                dtype=np.float64,
            )
            # The fitted center cancels exactly in a within-prompt difference.
            scores = (heldout_last - heldout_reference) @ detector["direction"]
            for capture, score in zip(np.asarray(captures, dtype=object)[test], scores):
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
        global_gap = min(relation_present) - max(relation_absent)
        eligible_gap = min(eligible_positive) - max(eligible_control)
        reports.append(
            {
                "reference": reference,
                "global_gap": global_gap,
                "eligible_gap": eligible_gap,
                "passes": global_gap > 0.0,
                "rows": rows,
            }
        )

    selected = next(
        (report["reference"] for report in reports if report["passes"]), None
    )
    return serializable(
        {
            "version": 1,
            "epistemic_status": (
                "post-closure architecture diagnostic on spent prompts; not "
                "calibration, certification, or holdout evidence"
            ),
            "layer": layer,
            "baseline_last_position_parity": parity,
            "direction_method": "raw_conjunction fitted LOCO on last positions",
            "reference_priority": list(REFERENCES),
            "selection_rule": "first reference with strictly positive global gap",
            "selected_reference": selected,
            "b_prime_survives": selected is not None,
            "references": reports,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=Path)
    parser.add_argument("--baseline-last-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        json.loads(args.captures.read_text()),
        json.loads(args.baseline_last_captures.read_text()),
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
