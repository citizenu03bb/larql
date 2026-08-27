#!/usr/bin/env python3
"""Evaluate one fixed relation-only detector on grouped spent captures.

This tool is for architecture selection, not certification.  It replaces the
learned entity factor with an oracle proxy for a future compiled token/alias
condition, fits only the relation direction, and reports coverage under a
fail-closed ternary decision.  It deliberately accepts one layer,
representation, method, and threshold policy per invocation: searching over
those choices belongs in an explicitly labelled exploratory outer driver.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_contextual_calibration import auc, best_threshold, unit, unit_rows
from analyze_contextual_positions import REPRESENTATIONS, matrix


METHODS = (
    "raw_conjunction",
    "unit_conjunction",
    "unit_factor_prototypes",
)
THRESHOLD_POLICIES = (
    "balanced",
    "zero_false_positive",
    "separation_gap",
)


def fit_relation(x: np.ndarray, cells: np.ndarray, method: str) -> dict:
    """Fit one relation direction while retaining all factorial supervision."""
    expected_cells = {"11", "10", "01", "00"}
    if set(cells) != expected_cells:
        raise ValueError(f"expected all factorial cells, got {sorted(set(cells))}")

    center = np.mean(x, axis=0)
    transformed = x - center
    if method.startswith("unit_"):
        transformed = unit_rows(transformed)
    means = {
        cell: np.mean(transformed[cells == cell], axis=0)
        for cell in expected_cells
    }
    if method.endswith("conjunction"):
        direction = unit(
            0.5 * ((means["11"] + means["01"]) - (means["10"] + means["00"]))
        )
    elif method == "unit_factor_prototypes":
        present = unit(
            np.mean(transformed[[cell[1] == "1" for cell in cells]], axis=0)
        )
        absent = unit(
            np.mean(transformed[[cell[1] == "0" for cell in cells]], axis=0)
        )
        direction = present - absent
    else:
        raise ValueError(f"unsupported method: {method}")
    return {"method": method, "center": center, "direction": direction}


def relation_scores(detector: dict, x: np.ndarray) -> np.ndarray:
    transformed = x - detector["center"]
    if detector["method"].startswith("unit_"):
        transformed = unit_rows(transformed)
    return transformed @ detector["direction"]


def select_thresholds(
    scores: np.ndarray, labels: np.ndarray, policy: str
) -> tuple[float, float]:
    """Return ``(no_fire_threshold, fire_threshold)`` from training data only."""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("threshold fitting requires both relation labels")
    if policy == "balanced":
        threshold = best_threshold(scores, labels)
        return threshold, threshold
    negative_ceiling = float(np.max(negatives))
    if policy == "zero_false_positive":
        return negative_ceiling, float(np.nextafter(negative_ceiling, np.inf))
    if policy == "separation_gap":
        positive_floor = float(np.min(positives))
        if positive_floor <= negative_ceiling:
            raise ValueError(
                "training relation scores have no strict separation gap: "
                f"negative_ceiling={negative_ceiling}, positive_floor={positive_floor}"
            )
        return negative_ceiling, positive_floor
    raise ValueError(f"unsupported threshold policy: {policy}")


def ternary_state(score: float, no_fire_threshold: float, fire_threshold: float) -> str:
    if score >= fire_threshold:
        return "fire"
    if score <= no_fire_threshold:
        return "no_fire"
    return "abstain"


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def partition_suites(
    suites: dict[str, list[dict]], group_field: str
) -> dict[str, list[dict]]:
    """Partition capture files by an explicit construction-group field."""
    grouped: dict[str, list[dict]] = {}
    for suite_name, captures in suites.items():
        for capture in captures:
            if group_field not in capture:
                raise ValueError(
                    f"capture {capture.get('name', '<unnamed>')!r} has no "
                    f"{group_field!r} field"
                )
            group_name = f"{suite_name}:{capture[group_field]}"
            grouped.setdefault(group_name, []).append(capture)
    return grouped


def primary_score(report: dict) -> float:
    """Scalarize the preregistered safety-first grouped outcome."""
    if report["invalid_folds"]:
        return -2.0
    aggregate = report["aggregate"]
    if aggregate["control_fires"] == 0:
        return float(aggregate["positive_fire_coverage"])
    return -float(aggregate["control_fires"]) / float(aggregate["control_n"])


def relation_flip_null(
    suites: dict[str, list[dict]],
    layer: int,
    representation: str,
    method: str,
    threshold_policy: str,
) -> dict:
    """Run the exact null formed by flipping the relation bit per group."""
    names = sorted(suites)
    if len(names) > 16:
        raise ValueError("exact relation-flip null is limited to 16 groups")
    scores = []
    observed = None
    for bits in itertools.product((0, 1), repeat=len(names)):
        permuted = {}
        for name, flip in zip(names, bits):
            rows = []
            for capture in suites[name]:
                row = dict(capture)
                if flip:
                    entity_bit, relation_bit = row["cell"]
                    row["cell"] = entity_bit + ("0" if relation_bit == "1" else "1")
                rows.append(row)
            permuted[name] = rows
        score_value = primary_score(
            evaluate(
                permuted,
                layer=layer,
                representation=representation,
                method=method,
                threshold_policy=threshold_policy,
            )
        )
        scores.append(score_value)
        if not any(bits):
            observed = score_value
    percentile_95 = float(np.quantile(scores, 0.95, method="higher"))
    return {
        "scheme": "exact per-group relation-bit flips",
        "n_permutations": len(scores),
        "observed_primary_score": observed,
        "null_95th_percentile": percentile_95,
        "passes": observed is not None and observed > percentile_95,
        "primary_scores": scores,
    }


def compile_calibrated_detector(
    development: list[dict],
    calibration: list[dict],
    layer: int,
    representation: str,
    method: str,
    minimum_normalized_gap: float,
) -> dict:
    """Fit once on development and calibrate a strict ternary gap once."""
    development_x = matrix(development, layer, representation)
    development_cells = np.asarray([row["cell"] for row in development])
    calibration_x = matrix(calibration, layer, representation)
    calibration_cells = np.asarray([row["cell"] for row in calibration])
    detector = fit_relation(development_x, development_cells, method)
    development_scores = relation_scores(detector, development_x)
    calibration_scores = relation_scores(detector, calibration_x)
    calibration_labels = np.asarray(
        [int(cell[1]) for cell in calibration_cells], dtype=np.int8
    )

    reasons = []
    try:
        no_fire_threshold, fire_threshold = select_thresholds(
            calibration_scores, calibration_labels, "separation_gap"
        )
    except ValueError as error:
        no_fire_threshold = None
        fire_threshold = None
        reasons.append(str(error))

    development_range = float(np.ptp(development_scores))
    gap = None
    normalized_gap = None
    rows = []
    if no_fire_threshold is not None and fire_threshold is not None:
        gap = fire_threshold - no_fire_threshold
        if development_range <= 0.0:
            reasons.append("development relation score range is not positive")
        else:
            normalized_gap = gap / development_range
            if normalized_gap < minimum_normalized_gap:
                reasons.append(
                    "calibration gap is below the preregistered normalized minimum: "
                    f"{normalized_gap} < {minimum_normalized_gap}"
                )
        for split, captures, scores in (
            ("development", development, development_scores),
            ("calibration", calibration, calibration_scores),
        ):
            for capture, score_value in zip(captures, scores):
                entity_present = capture["cell"][0] == "1"
                relation_present = capture["cell"][1] == "1"
                state = (
                    ternary_state(
                        float(score_value), no_fire_threshold, fire_threshold
                    )
                    if entity_present
                    else "compiled_reject"
                )
                rows.append(
                    {
                        "split": split,
                        "name": capture["name"],
                        "prompt": capture["prompt"],
                        "cell": capture["cell"],
                        "score": float(score_value),
                        "state": state,
                    }
                )
                if entity_present and relation_present and state != "fire":
                    reasons.append(
                        f"supported positive is not fire: {split}/{capture['name']}={state}"
                    )
                if entity_present and not relation_present and state == "fire":
                    reasons.append(
                        f"entity-only control fires: {split}/{capture['name']}"
                    )

    passed = not reasons
    report = {
        "version": 1,
        "epistemic_status": "prospective development fit and calibration",
        "candidate": {
            "layer": layer,
            "representation": representation,
            "method": method,
            "threshold_policy": "separation_gap",
            "minimum_normalized_gap": minimum_normalized_gap,
        },
        "passed": passed,
        "reasons": reasons,
        "metrics": {
            "development_range": development_range,
            "calibration_gap": gap,
            "normalized_calibration_gap": normalized_gap,
        },
        "rows": rows,
        "detector": None,
    }
    if passed:
        report["detector"] = {
            "center": detector["center"],
            "relation_direction": detector["direction"],
            "no_fire_threshold": no_fire_threshold,
            "fire_threshold": fire_threshold,
        }
    return serializable(report)


def evaluate(
    suites: dict[str, list[dict]],
    layer: int,
    representation: str,
    method: str,
    threshold_policy: str,
) -> dict:
    if len(suites) < 3:
        raise ValueError("expected at least three named suites")
    matrices = {
        name: matrix(captures, layer, representation)
        for name, captures in suites.items()
    }
    cells = {
        name: np.asarray([capture["cell"] for capture in captures])
        for name, captures in suites.items()
    }

    rows = []
    suite_metrics = []
    invalid_folds = []
    for heldout_name, heldout in suites.items():
        train_names = [name for name in suites if name != heldout_name]
        train_x = np.concatenate([matrices[name] for name in train_names])
        train_cells = np.concatenate([cells[name] for name in train_names])
        detector = fit_relation(train_x, train_cells, method)
        train_scores = relation_scores(detector, train_x)
        train_labels = np.asarray(
            [int(cell[1]) for cell in train_cells], dtype=np.int8
        )
        try:
            no_fire_threshold, fire_threshold = select_thresholds(
                train_scores, train_labels, threshold_policy
            )
        except ValueError as error:
            invalid_folds.append({"suite": heldout_name, "reason": str(error)})
            continue

        heldout_scores = relation_scores(detector, matrices[heldout_name])
        heldout_rows = []
        for capture, score_value in zip(heldout, heldout_scores):
            score_value = float(score_value)
            entity_present = capture["cell"][0] == "1"
            relation_present = capture["cell"][1] == "1"
            state = (
                ternary_state(score_value, no_fire_threshold, fire_threshold)
                if entity_present
                else "compiled_reject"
            )
            row = {
                "suite": heldout_name,
                "name": capture["name"],
                "prompt": capture["prompt"],
                "cell": capture["cell"],
                "entity_oracle_present": entity_present,
                "relation_present": relation_present,
                "relation_score": score_value,
                "no_fire_threshold": no_fire_threshold,
                "fire_threshold": fire_threshold,
                "state": state,
            }
            rows.append(row)
            heldout_rows.append(row)

        eligible = [row for row in heldout_rows if row["entity_oracle_present"]]
        positives = [row for row in eligible if row["relation_present"]]
        controls = [row for row in eligible if not row["relation_present"]]
        suite_metrics.append(
            {
                "suite": heldout_name,
                "positive_fire_coverage": sum(
                    row["state"] == "fire" for row in positives
                )
                / len(positives),
                "positive_fires": sum(row["state"] == "fire" for row in positives),
                "positive_n": len(positives),
                "control_fires": sum(row["state"] == "fire" for row in controls),
                "control_n": len(controls),
                "abstentions": sum(row["state"] == "abstain" for row in eligible),
                "worst_positive_fire_margin": min(
                    row["relation_score"] - row["fire_threshold"]
                    for row in positives
                ),
                "worst_control_safety_margin": min(
                    row["fire_threshold"] - row["relation_score"]
                    for row in controls
                ),
            }
        )

    eligible = [row for row in rows if row["entity_oracle_present"]]
    positives = [row for row in eligible if row["relation_present"]]
    controls = [row for row in eligible if not row["relation_present"]]
    labels = np.asarray([int(row["relation_present"]) for row in eligible])
    scores = np.asarray([row["relation_score"] for row in eligible])
    report = {
        "version": 1,
        "epistemic_status": (
            "exploratory architecture selection on spent captures; not "
            "certification, calibration, or holdout evidence"
        ),
        "protocol": "leave-one-entire-suite-out",
        "candidate": {
            "layer": layer,
            "representation": representation,
            "method": method,
            "threshold_policy": threshold_policy,
        },
        "contract": {
            "entity_gate": (
                "cell entity bit used only as an oracle proxy for a future "
                "compiled token/alias condition"
            ),
            "relation_gate": "learned from all four factorial cells",
            "states": ["fire", "abstain", "no_fire", "compiled_reject"],
            "primary_metric": "positive fire coverage under zero control fires",
        },
        "suite_sizes": {name: len(captures) for name, captures in suites.items()},
        "invalid_folds": invalid_folds,
        "suite_metrics": suite_metrics,
        "aggregate": {
            "eligible_n": len(eligible),
            "positive_n": len(positives),
            "positive_fires": sum(row["state"] == "fire" for row in positives),
            "positive_fire_coverage": (
                sum(row["state"] == "fire" for row in positives) / len(positives)
                if positives
                else 0.0
            ),
            "control_n": len(controls),
            "control_fires": sum(row["state"] == "fire" for row in controls),
            "abstentions": sum(row["state"] == "abstain" for row in eligible),
            "auc": auc(labels, scores) if positives and controls else None,
            "worst_positive_fire_margin": (
                min(
                    row["relation_score"] - row["fire_threshold"]
                    for row in positives
                )
                if positives
                else None
            ),
            "worst_control_safety_margin": (
                min(
                    row["fire_threshold"] - row["relation_score"]
                    for row in controls
                )
                if controls
                else None
            ),
        },
        "rows": rows,
    }
    return serializable(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="+", help="NAME=PATH capture files")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--threshold-policy", choices=THRESHOLD_POLICIES, required=True
    )
    parser.add_argument(
        "--group-field",
        help="partition each input capture file by this construction-group field",
    )
    parser.add_argument(
        "--relation-flip-null",
        action="store_true",
        help="attach the exact per-group relation-bit-flip null distribution",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    suites = {}
    for item in args.suites:
        name, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"suite must be NAME=PATH, got {item!r}")
        suites[name] = json.loads(Path(path).read_text())
    if args.group_field:
        suites = partition_suites(suites, args.group_field)
    report = evaluate(
        suites,
        layer=args.layer,
        representation=args.representation,
        method=args.method,
        threshold_policy=args.threshold_policy,
    )
    if args.relation_flip_null:
        report["relation_flip_null"] = relation_flip_null(
            suites,
            layer=args.layer,
            representation=args.representation,
            method=args.method,
            threshold_policy=args.threshold_policy,
        )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
