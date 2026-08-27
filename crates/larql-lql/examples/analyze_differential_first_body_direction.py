#!/usr/bin/env python3
"""Evaluate one exploratory relation direction learned on first-body deltas."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_factorized_relation import (
    fit_relation,
    relation_scores,
    serializable,
)
from analyze_contextual_positions import positions_at
from analyze_internal_reference_relation import (
    validate_factorial_captures,
    verify_last_position_parity,
)


def first_body_delta(capture: dict, layer: int) -> np.ndarray:
    positions = positions_at(capture, layer)
    if len(positions) != len(capture["prompt_ids"]):
        raise ValueError(
            "position residual count differs from prompt token count: "
            f"{len(positions)} != {len(capture['prompt_ids'])}"
        )
    if len(positions) < 3:
        raise ValueError("first-body delta requires BOS, body, and final positions")
    return positions[-1] - positions[1]


def verify_matched_first_tokens(captures: list[dict]) -> dict:
    rows = []
    for fold in range(6):
        for entity_bit in ("1", "0"):
            present = next(
                capture
                for capture in captures
                if int(capture["fold"]) == fold
                and capture["cell"] == entity_bit + "1"
            )
            absent = next(
                capture
                for capture in captures
                if int(capture["fold"]) == fold
                and capture["cell"] == entity_bit + "0"
            )
            present_token = int(present["prompt_ids"][1])
            absent_token = int(absent["prompt_ids"][1])
            if present_token != absent_token:
                raise ValueError(
                    "analysis-inconclusive: matched pair has different first "
                    f"body tokens for fold={fold}, entity_bit={entity_bit}: "
                    f"{present_token} != {absent_token}"
                )
            rows.append(
                {
                    "fold": fold,
                    "entity_bit": entity_bit,
                    "token_id": present_token,
                }
            )
    return {"status": "exact", "matched_pairs": len(rows), "rows": rows}


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


def analyze(captures: list[dict], baseline: list[dict], layer: int = 13) -> dict:
    folds = validate_factorial_captures(captures)
    parity = verify_last_position_parity(captures, baseline, layer)
    prefix_parity = verify_matched_first_tokens(captures)
    cells = np.asarray([capture["cell"] for capture in captures])
    capture_folds = np.asarray([int(capture["fold"]) for capture in captures])
    z = np.asarray(
        [first_body_delta(capture, layer) for capture in captures],
        dtype=np.float64,
    )
    capture_array = np.asarray(captures, dtype=object)

    rows = []
    directions = []
    for heldout in folds:
        train = capture_folds != heldout
        test = ~train
        detector = fit_relation(z[train], cells[train], "raw_conjunction")
        if not np.all(np.isfinite(detector["direction"])):
            raise ValueError("analysis-inconclusive: non-finite differential direction")
        if float(np.linalg.norm(detector["direction"])) == 0.0:
            raise ValueError("analysis-inconclusive: zero differential direction")
        directions.append(detector["direction"])
        scores = relation_scores(detector, z[test])
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
    global_gap = min(relation_present) - max(relation_absent)
    eligible_gap = min(eligible_positive) - max(eligible_control)
    paired = paired_score_deltas(rows)
    cosines = direction_cosines(directions)

    return serializable(
        {
            "version": 1,
            "epistemic_status": (
                "post-hoc exploratory nomination test on spent prompts; not "
                "calibration, prospective evidence, certification, or holdout evidence"
            ),
            "layer": layer,
            "baseline_last_position_parity": parity,
            "matched_first_body_tokens": prefix_parity,
            "representation": "x_last minus x_first_body",
            "direction_method": "raw_conjunction fitted LOCO on differential vectors",
            "global_gap": global_gap,
            "eligible_gap": eligible_gap,
            "passes": global_gap > 0.0,
            "minimum_paired_delta": min(row["delta"] for row in paired),
            "paired_score_deltas": paired,
            "minimum_direction_cosine": min(row["cosine"] for row in cosines),
            "direction_cosines": cosines,
            "rows": rows,
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
