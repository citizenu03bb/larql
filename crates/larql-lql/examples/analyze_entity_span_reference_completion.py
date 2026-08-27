#!/usr/bin/env python3
"""Complete the single predeclared B-prime entity-span reference arm.

This analyzer changes only the alias table which made the original arm
non-executable.  It retains the last-position raw-conjunction direction and
evaluates one entity-span reference on the spent six-construction capture.
The result is exploratory and cannot certify or calibrate a detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_factorized_relation import fit_relation, serializable
from analyze_contextual_positions import positions_at
from analyze_internal_reference_relation import (
    validate_factorial_captures,
    verify_last_position_parity,
)


ENTITY_ALIASES = (
    ("archive_zephyr_start", (39140, 7733, 109999)),
    ("archive_zephyr_space", (40187, 7733, 109999)),
    ("archive_nimbus_start", (39140, 129201)),
    ("archive_nimbus_space", (40187, 129201)),
)


def exact_entity_span(prompt_ids: list[int]) -> tuple[str, int, int]:
    matches = []
    for alias_name, alias_ids in ENTITY_ALIASES:
        for start in range(len(prompt_ids)):
            end = start + len(alias_ids)
            if tuple(prompt_ids[start:end]) == alias_ids:
                matches.append((alias_name, start, end))
    if not matches:
        raise ValueError("prompt contains no frozen Archive entity alias")
    if len(matches) != 1:
        raise ValueError(f"prompt must contain exactly one entity alias: {matches}")
    return matches[0]


def entity_reference(capture: dict, layer: int) -> tuple[str, np.ndarray]:
    alias_name, start, end = exact_entity_span(capture["prompt_ids"])
    positions = positions_at(capture, layer)
    if len(positions) != len(capture["prompt_ids"]):
        raise ValueError(
            "position residual count differs from prompt token count: "
            f"{len(positions)} != {len(capture['prompt_ids'])}"
        )
    return alias_name, np.mean(positions[start:end], axis=0)


def paired_diagnostics(rows: list[dict]) -> dict:
    score_deltas = []
    reference_projection_deltas = []
    prompt_length_deltas = []
    for fold in range(6):
        for entity_bit in ("1", "0"):
            present_cell = entity_bit + "1"
            absent_cell = entity_bit + "0"
            present = next(
                row
                for row in rows
                if row["fold"] == fold and row["cell"] == present_cell
            )
            absent = next(
                row
                for row in rows
                if row["fold"] == fold and row["cell"] == absent_cell
            )
            score_deltas.append(
                {
                    "fold": fold,
                    "entity_bit": entity_bit,
                    "delta": present["score"] - absent["score"],
                }
            )
            reference_projection_deltas.append(
                {
                    "fold": fold,
                    "entity_bit": entity_bit,
                    "delta": (
                        present["reference_projection"]
                        - absent["reference_projection"]
                    ),
                }
            )
            prompt_length_deltas.append(
                {
                    "fold": fold,
                    "entity_bit": entity_bit,
                    "delta": present["prompt_token_count"]
                    - absent["prompt_token_count"],
                }
            )
    return {
        "minimum_score_delta": min(row["delta"] for row in score_deltas),
        "maximum_absolute_reference_projection_delta": max(
            abs(row["delta"]) for row in reference_projection_deltas
        ),
        "score_deltas": score_deltas,
        "reference_projection_deltas": reference_projection_deltas,
        "prompt_length_deltas": prompt_length_deltas,
    }


def analyze(captures: list[dict], baseline: list[dict], layer: int = 13) -> dict:
    folds = validate_factorial_captures(captures)
    parity = verify_last_position_parity(captures, baseline, layer)
    cells = np.asarray([capture["cell"] for capture in captures])
    capture_folds = np.asarray([int(capture["fold"]) for capture in captures])
    last = np.asarray(
        [positions_at(capture, layer)[-1] for capture in captures],
        dtype=np.float64,
    )
    capture_array = np.asarray(captures, dtype=object)

    rows = []
    for heldout in folds:
        train = capture_folds != heldout
        test = ~train
        detector = fit_relation(last[train], cells[train], "raw_conjunction")
        for capture, last_vector in zip(capture_array[test], last[test]):
            alias_name, reference = entity_reference(capture, layer)
            rows.append(
                {
                    "fold": heldout,
                    "name": capture["name"],
                    "prompt": capture["prompt"],
                    "prompt_token_count": len(capture["prompt_ids"]),
                    "cell": capture["cell"],
                    "entity_alias": alias_name,
                    "score": float(
                        (last_vector - reference) @ detector["direction"]
                    ),
                    "reference_projection": float(
                        reference @ detector["direction"]
                    ),
                }
            )

    relation_present = [row["score"] for row in rows if row["cell"][1] == "1"]
    relation_absent = [row["score"] for row in rows if row["cell"][1] == "0"]
    eligible_positive = [row["score"] for row in rows if row["cell"] == "11"]
    eligible_control = [row["score"] for row in rows if row["cell"] == "10"]
    global_gap = min(relation_present) - max(relation_absent)
    eligible_gap = min(eligible_positive) - max(eligible_control)

    return serializable(
        {
            "version": 1,
            "epistemic_status": (
                "exploratory mechanical completion of the predeclared "
                "entity-span arm; not calibration, certification, or holdout evidence"
            ),
            "layer": layer,
            "baseline_last_position_parity": parity,
            "direction_method": "raw_conjunction fitted LOCO on last positions",
            "reference": "mean residual over first exact frozen entity alias",
            "alias_names": [name for name, _ in ENTITY_ALIASES],
            "global_gap": global_gap,
            "eligible_gap": eligible_gap,
            "passes": global_gap > 0.0,
            "paired": paired_diagnostics(rows),
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
