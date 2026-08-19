#!/usr/bin/env python3
"""Suite-level calibration study for contextual multi-token detectors.

Every reported score is out-of-suite: the complete named suite containing a
row is excluded while fitting its layer, directions, and thresholds.  The
script deliberately evaluates only detector families that can be represented
by a small number of stored vectors and dot products at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = (
    "raw_single",
    "unit_single",
    "raw_conjunction",
    "unit_conjunction",
    "unit_factor_prototypes",
    "unit_cell_prototypes",
)
ENSEMBLE_METHODS = (
    "raw_conjunction",
    "unit_conjunction",
    "unit_factor_prototypes",
)
ENSEMBLE_WIDTHS = (3, 5)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm == 0.0 else vector / norm


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.unique(scores)
    if len(values) == 1:
        return float(values[0])
    candidates = np.concatenate(
        ([values[0] - 1e-6], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-6])
    )
    best_key = None
    selected = float(candidates[0])
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = float(np.mean(predicted[labels == 1]))
        specificity = float(np.mean(~predicted[labels == 0]))
        key = (0.5 * (sensitivity + specificity), min(sensitivity, specificity))
        if best_key is None or key > best_key:
            best_key = key
            selected = float(threshold)
    return selected


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / (len(positives) * len(negatives))


def layer_matrix(captures: list[dict], layer: int) -> np.ndarray:
    rows = []
    for capture in captures:
        residuals = {item[0]: item[1] for item in capture["residuals"]}
        rows.append(residuals[layer])
    return np.asarray(rows, dtype=np.float64)


def fit_detector(captures: list[dict], layer: int, method: str) -> dict:
    x = layer_matrix(captures, layer)
    cells = np.asarray([capture["cell"] for capture in captures])
    center = np.mean(x, axis=0)
    transformed = x - center
    if method.startswith("unit_"):
        transformed = unit_rows(transformed)
    means = {
        cell: np.mean(transformed[cells == cell], axis=0)
        for cell in ("11", "10", "01", "00")
    }
    detector = {"method": method, "layer": layer, "center": center}

    if method.endswith("single"):
        direction = unit(means["11"] - 0.5 * (means["10"] + means["01"]))
        labels = (cells == "11").astype(np.int8)
        scores = transformed @ direction
        detector.update(
            direction=direction,
            threshold=best_threshold(scores, labels),
        )
    elif method.endswith("conjunction"):
        entity_direction = unit(
            0.5 * ((means["11"] + means["10"]) - (means["01"] + means["00"]))
        )
        relation_direction = unit(
            0.5 * ((means["11"] + means["01"]) - (means["10"] + means["00"]))
        )
        entity_labels = np.asarray([int(cell[0]) for cell in cells], dtype=np.int8)
        relation_labels = np.asarray([int(cell[1]) for cell in cells], dtype=np.int8)
        detector.update(
            entity_direction=entity_direction,
            relation_direction=relation_direction,
            entity_threshold=best_threshold(
                transformed @ entity_direction, entity_labels
            ),
            relation_threshold=best_threshold(
                transformed @ relation_direction, relation_labels
            ),
        )
    elif method == "unit_factor_prototypes":
        entity_present = unit(np.mean(transformed[[cell[0] == "1" for cell in cells]], axis=0))
        entity_absent = unit(np.mean(transformed[[cell[0] == "0" for cell in cells]], axis=0))
        relation_present = unit(
            np.mean(transformed[[cell[1] == "1" for cell in cells]], axis=0)
        )
        relation_absent = unit(
            np.mean(transformed[[cell[1] == "0" for cell in cells]], axis=0)
        )
        detector.update(
            entity_present=entity_present,
            entity_absent=entity_absent,
            relation_present=relation_present,
            relation_absent=relation_absent,
        )
    elif method == "unit_cell_prototypes":
        detector["prototypes"] = {cell: unit(mean) for cell, mean in means.items()}
    else:
        raise ValueError(f"unknown method: {method}")
    return detector


def score_detector(detector: dict, captures: list[dict]) -> np.ndarray:
    transformed = layer_matrix(captures, detector["layer"]) - detector["center"]
    method = detector["method"]
    if method.startswith("unit_"):
        transformed = unit_rows(transformed)
    if method.endswith("single"):
        return transformed @ detector["direction"] - detector["threshold"]
    if method.endswith("conjunction"):
        return np.minimum(
            transformed @ detector["entity_direction"]
            - detector["entity_threshold"],
            transformed @ detector["relation_direction"]
            - detector["relation_threshold"],
        )
    if method == "unit_factor_prototypes":
        entity = transformed @ detector["entity_present"] - transformed @ detector["entity_absent"]
        relation = (
            transformed @ detector["relation_present"]
            - transformed @ detector["relation_absent"]
        )
        return np.minimum(entity, relation)
    if method == "unit_cell_prototypes":
        similarities = {
            cell: transformed @ prototype
            for cell, prototype in detector["prototypes"].items()
        }
        return similarities["11"] - np.maximum.reduce(
            [similarities[cell] for cell in ("10", "01", "00")]
        )
    raise ValueError(f"unknown method: {method}")


def serializable_detector(detector: dict) -> dict:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.astype(np.float32).tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return {key: convert(value) for key, value in detector.items()}


def fit_ensemble(captures: list[dict], layers: list[int], method: str) -> dict:
    members = []
    for layer in layers:
        detector = fit_detector(captures, layer, method)
        train_scores = score_detector(detector, captures)
        scale = float(np.std(train_scores))
        members.append({"scale": scale if scale > 0 else 1.0, "detector": detector})
    return {"method": f"ensemble:{method}", "layers": layers, "members": members}


def score_ensemble(ensemble: dict, captures: list[dict]) -> np.ndarray:
    return np.mean(
        [
            score_detector(member["detector"], captures) / member["scale"]
            for member in ensemble["members"]
        ],
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suites",
        nargs="+",
        help="named capture files as NAME=PATH (at least three)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-output", type=Path)
    args = parser.parse_args()

    suites: dict[str, list[dict]] = {}
    for item in args.suites:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"suite must be NAME=PATH, got {item!r}")
        suites[name] = json.loads(Path(path).read_text())
    if len(suites) < 3:
        raise ValueError("expected at least three suites")
    expected_cells = {"11", "10", "01", "00"}
    for name, captures in suites.items():
        if {capture["cell"] for capture in captures} != expected_cells:
            raise ValueError(f"{name}: expected all four factorial cells")

    all_captures = [capture for captures in suites.values() for capture in captures]
    layers = sorted(
        set.intersection(
            *[
                {item[0] for item in capture["residuals"]}
                for capture in all_captures
            ]
        )
    )
    candidates = []
    for layer in layers:
        for method in METHODS:
            rows = []
            suite_metrics = []
            for heldout_name, heldout in suites.items():
                train = [
                    capture
                    for name, captures in suites.items()
                    if name != heldout_name
                    for capture in captures
                ]
                detector = fit_detector(train, layer, method)
                scores = score_detector(detector, heldout)
                labels = np.asarray(
                    [capture["cell"] == "11" for capture in heldout], dtype=np.int8
                )
                signed = np.where(labels == 1, scores, -scores)
                suite_metrics.append(
                    {
                        "suite": heldout_name,
                        "correct": int(np.sum(signed >= 0)),
                        "n": len(heldout),
                        "decision_margin": float(np.min(signed)),
                        "ranking_margin": float(
                            np.min(scores[labels == 1]) - np.max(scores[labels == 0])
                        ),
                    }
                )
                rows.extend(
                    {
                        "suite": heldout_name,
                        "name": capture["name"],
                        "cell": capture["cell"],
                        "score": float(score),
                        "correct": bool(sign >= 0),
                    }
                    for capture, score, sign in zip(heldout, scores, signed)
                )
            labels = np.asarray([row["cell"] == "11" for row in rows], dtype=np.int8)
            scores = np.asarray([row["score"] for row in rows])
            signed = np.where(labels == 1, scores, -scores)
            candidates.append(
                {
                    "layer": layer,
                    "method": method,
                    "correct": int(np.sum(signed >= 0)),
                    "n": len(rows),
                    "decision_margin": float(np.min(signed)),
                    "ranking_margin": float(
                        np.min(scores[labels == 1]) - np.max(scores[labels == 0])
                    ),
                    "auc": auc(labels, scores),
                    "suite_metrics": suite_metrics,
                    "rows": rows,
                }
            )

    for width in ENSEMBLE_WIDTHS:
        for start in range(len(layers) - width + 1):
            window = layers[start : start + width]
            if any(right != left + 1 for left, right in zip(window, window[1:])):
                continue
            for base_method in ENSEMBLE_METHODS:
                rows = []
                suite_metrics = []
                for heldout_name, heldout in suites.items():
                    train = [
                        capture
                        for name, captures in suites.items()
                        if name != heldout_name
                        for capture in captures
                    ]
                    ensemble = fit_ensemble(train, window, base_method)
                    scores = score_ensemble(ensemble, heldout)
                    labels = np.asarray(
                        [capture["cell"] == "11" for capture in heldout],
                        dtype=np.int8,
                    )
                    signed = np.where(labels == 1, scores, -scores)
                    suite_metrics.append(
                        {
                            "suite": heldout_name,
                            "correct": int(np.sum(signed >= 0)),
                            "n": len(heldout),
                            "decision_margin": float(np.min(signed)),
                            "ranking_margin": float(
                                np.min(scores[labels == 1])
                                - np.max(scores[labels == 0])
                            ),
                        }
                    )
                    rows.extend(
                        {
                            "suite": heldout_name,
                            "name": capture["name"],
                            "cell": capture["cell"],
                            "score": float(score),
                            "correct": bool(sign >= 0),
                        }
                        for capture, score, sign in zip(heldout, scores, signed)
                    )
                labels = np.asarray(
                    [row["cell"] == "11" for row in rows], dtype=np.int8
                )
                scores = np.asarray([row["score"] for row in rows])
                signed = np.where(labels == 1, scores, -scores)
                candidates.append(
                    {
                        "layers": window,
                        "method": f"ensemble:{base_method}",
                        "correct": int(np.sum(signed >= 0)),
                        "n": len(rows),
                        "decision_margin": float(np.min(signed)),
                        "ranking_margin": float(
                            np.min(scores[labels == 1])
                            - np.max(scores[labels == 0])
                        ),
                        "auc": auc(labels, scores),
                        "suite_metrics": suite_metrics,
                        "rows": rows,
                    }
                )

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["correct"],
            min(item["correct"] / item["n"] for item in candidate["suite_metrics"]),
            candidate["decision_margin"],
            candidate["ranking_margin"],
        ),
    )
    perfect = [
        candidate
        for candidate in candidates
        if candidate["correct"] == candidate["n"]
        and candidate["decision_margin"] > 0
    ]
    report = {
        "protocol": "leave-one-entire-suite-out",
        "suite_sizes": {name: len(captures) for name, captures in suites.items()},
        "n_captures": len(all_captures),
        "layers": layers,
        "methods": list(METHODS),
        "ensemble_methods": list(ENSEMBLE_METHODS),
        "ensemble_widths": list(ENSEMBLE_WIDTHS),
        "acceptance": "32/32 decisions and positive worst signed margin",
        "accepted_candidates": len(perfect),
        "best": best,
        "top_candidates": [
            {key: value for key, value in candidate.items() if key != "rows"}
            for candidate in sorted(
                candidates,
                key=lambda candidate: (
                    candidate["correct"],
                    candidate["decision_margin"],
                    candidate["ranking_margin"],
                ),
                reverse=True,
            )[:20]
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    if args.detector_output:
        if not perfect:
            raise ValueError("no candidate met acceptance; refusing to freeze a detector")
        selected = max(
            perfect,
            key=lambda candidate: (
                candidate["decision_margin"], candidate["ranking_margin"]
            ),
        )
        if "layers" in selected:
            base_method = selected["method"].removeprefix("ensemble:")
            detector = fit_ensemble(all_captures, selected["layers"], base_method)
        else:
            detector = fit_detector(all_captures, selected["layer"], selected["method"])
        frozen = {
            "version": 1,
            "selection_protocol": "leave-one-entire-suite-out on three development suites",
            "selection_metrics": {
                key: value
                for key, value in selected.items()
                if key not in ("rows", "suite_metrics")
            },
            **serializable_detector(detector),
        }
        args.detector_output.write_text(json.dumps(frozen) + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
