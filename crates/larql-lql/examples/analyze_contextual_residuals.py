#!/usr/bin/env python3
"""Offline factor analysis for multitoken_insert_baseline residual captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    wins = 0.0
    for positive in positives:
        wins += np.sum(positive > negatives)
        wins += 0.5 * np.sum(positive == negatives)
    return float(wins / (len(positives) * len(negatives)))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.unique(scores)
    if len(values) == 1:
        return float(values[0])
    candidates = np.concatenate(
        ([values[0] - 1e-6], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-6])
    )
    best = (float("-inf"), float(candidates[0]))
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = np.mean(predicted[labels == 1])
        true_negative = np.mean(~predicted[labels == 0])
        balanced = 0.5 * (true_positive + true_negative)
        if balanced > best[0]:
            best = (float(balanced), float(threshold))
    return best[1]


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def margin(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.min(scores[labels == 1]) - np.max(scores[labels == 0]))


def layer_matrix(captures: list[dict], layer: int) -> np.ndarray:
    rows = []
    for capture in captures:
        residual_by_layer = {item[0]: item[1] for item in capture["residuals"]}
        rows.append(residual_by_layer[layer])
    return np.asarray(rows, dtype=np.float64)


def evaluate_layer(
    captures: list[dict], layer: int, random_trials: int, seed: int
) -> dict:
    x = layer_matrix(captures, layer)
    cells = np.asarray([capture["cell"] for capture in captures])
    folds = np.asarray([capture["fold"] for capture in captures])
    primary = np.isin(cells, ["11", "10", "01"])
    labels = (cells == "11").astype(np.int8)
    entity_labels = np.asarray([int(cell[0]) for cell in cells], dtype=np.int8)
    relation_labels = np.asarray([int(cell[1]) for cell in cells], dtype=np.int8)

    score_a = np.zeros(len(captures), dtype=np.float64)
    score_b = np.zeros(len(captures), dtype=np.float64)
    interaction_norms = []
    random_scores = np.zeros((random_trials, len(captures)), dtype=np.float64)
    rng = np.random.default_rng(seed + layer)

    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        center = np.mean(x[train], axis=0)
        train_x = x[train] - center
        test_x = x[test] - center

        means = {
            cell: np.mean(train_x[cells[train] == cell], axis=0)
            for cell in ("11", "10", "01", "00")
        }
        direction_entity = unit(
            0.5 * ((means["11"] + means["10"]) - (means["01"] + means["00"]))
        )
        direction_relation = unit(
            0.5 * ((means["11"] + means["01"]) - (means["10"] + means["00"]))
        )
        direction_single = unit(
            means["11"] - 0.5 * (means["10"] + means["01"])
        )
        interaction_norms.append(
            float(
                np.linalg.norm(
                    means["11"] - means["10"] - means["01"] + means["00"]
                )
            )
        )

        entity_train_scores = train_x @ direction_entity
        relation_train_scores = train_x @ direction_relation
        entity_threshold = best_threshold(
            entity_train_scores, entity_labels[train]
        )
        relation_threshold = best_threshold(
            relation_train_scores, relation_labels[train]
        )
        entity_test_scores = test_x @ direction_entity - entity_threshold
        relation_test_scores = test_x @ direction_relation - relation_threshold
        score_a[test] = np.minimum(entity_test_scores, relation_test_scores)
        score_b[test] = test_x @ direction_single

        for trial in range(random_trials):
            direction_random = unit(rng.standard_normal(x.shape[1]))
            random_scores[trial, test] = test_x @ direction_random

    primary_labels = labels[primary]
    auc_random = np.asarray(
        [auc(primary_labels, scores[primary]) for scores in random_scores]
    )
    auc_random_all = np.asarray([auc(labels, scores) for scores in random_scores])
    return {
        "layer": layer,
        "auc_conjunction": auc(primary_labels, score_a[primary]),
        "auc_single": auc(primary_labels, score_b[primary]),
        "margin_conjunction": margin(primary_labels, score_a[primary]),
        "margin_single": margin(primary_labels, score_b[primary]),
        "random_auc_p95": float(np.quantile(auc_random, 0.95)),
        "auc_conjunction_all_negatives": auc(labels, score_a),
        "auc_single_all_negatives": auc(labels, score_b),
        "margin_conjunction_all_negatives": margin(labels, score_a),
        "margin_single_all_negatives": margin(labels, score_b),
        "random_auc_all_negatives_p95": float(np.quantile(auc_random_all, 0.95)),
        "interaction_norm_mean": float(np.mean(interaction_norms)),
        "scores": [
            {
                "name": capture["name"],
                "cell": capture["cell"],
                "fold": capture["fold"],
                "conjunction": float(score_a[index]),
                "single": float(score_b[index]),
            }
            for index, capture in enumerate(captures)
        ],
    }


def freeze_single_detector(captures: list[dict], layer: int) -> dict:
    x = layer_matrix(captures, layer)
    cells = np.asarray([capture["cell"] for capture in captures])
    labels = (cells == "11").astype(np.int8)
    center = np.mean(x, axis=0)
    centered = x - center
    means = {
        cell: np.mean(centered[cells == cell], axis=0)
        for cell in ("11", "10", "01", "00")
    }
    direction = unit(means["11"] - 0.5 * (means["10"] + means["01"]))
    scores = centered @ direction
    observed_margin = margin(labels, scores)
    if observed_margin > 0:
        threshold = 0.5 * (
            np.min(scores[labels == 1]) + np.max(scores[labels == 0])
        )
    else:
        threshold = best_threshold(scores, labels)
    return {
        "version": 1,
        "layer": layer,
        "construction": "unit(mu_11 - 0.5*(mu_10 + mu_01)); centered on all 12 discovery prompts",
        "threshold": float(threshold),
        "discovery_margin_all_negatives": observed_margin,
        "center": center.astype(np.float32).tolist(),
        "direction": direction.astype(np.float32).tolist(),
        "scores": [
            {
                "name": capture["name"],
                "cell": capture["cell"],
                "score": float(score),
                "passes": bool(score >= threshold),
            }
            for capture, score in zip(captures, scores)
        ],
    }


def freeze_conjunction_detector(captures: list[dict], layer: int) -> dict:
    x = layer_matrix(captures, layer)
    cells = np.asarray([capture["cell"] for capture in captures])
    center = np.mean(x, axis=0)
    centered = x - center
    means = {
        cell: np.mean(centered[cells == cell], axis=0)
        for cell in ("11", "10", "01", "00")
    }
    entity_direction = unit(
        0.5 * ((means["11"] + means["10"]) - (means["01"] + means["00"]))
    )
    relation_direction = unit(
        0.5 * ((means["11"] + means["01"]) - (means["10"] + means["00"]))
    )
    entity_labels = np.asarray([int(cell[0]) for cell in cells], dtype=np.int8)
    relation_labels = np.asarray([int(cell[1]) for cell in cells], dtype=np.int8)
    entity_scores = centered @ entity_direction
    relation_scores = centered @ relation_direction
    entity_threshold = best_threshold(entity_scores, entity_labels)
    relation_threshold = best_threshold(relation_scores, relation_labels)
    scores = np.minimum(
        entity_scores - entity_threshold,
        relation_scores - relation_threshold,
    )
    labels = (cells == "11").astype(np.int8)
    return {
        "version": 1,
        "kind": "conjunction",
        "layer": layer,
        "construction": (
            "min(entity projection - threshold, relation projection - threshold); "
            "centered and fitted on the 12 discovery prompts only"
        ),
        "entity_threshold": float(entity_threshold),
        "relation_threshold": float(relation_threshold),
        "discovery_margin_all_negatives": margin(labels, scores),
        "center": center.astype(np.float32).tolist(),
        "entity_direction": entity_direction.astype(np.float32).tolist(),
        "relation_direction": relation_direction.astype(np.float32).tolist(),
        "scores": [
            {
                "name": capture["name"],
                "cell": capture["cell"],
                "score": float(score),
                "passes": bool(score >= 0),
            }
            for capture, score in zip(captures, scores)
        ],
    }


def fixed_validation_metrics(
    discovery: list[dict], validation: list[dict], layer: int
) -> dict:
    """Exploratory only: train on discovery and score an already-opened validation set."""
    train_x = layer_matrix(discovery, layer)
    validation_x = layer_matrix(validation, layer)
    train_cells = np.asarray([capture["cell"] for capture in discovery])
    validation_cells = np.asarray([capture["cell"] for capture in validation])
    validation_labels = (validation_cells == "11").astype(np.int8)
    center = np.mean(train_x, axis=0)
    train_x = train_x - center
    validation_x = validation_x - center
    means = {
        cell: np.mean(train_x[train_cells == cell], axis=0)
        for cell in ("11", "10", "01", "00")
    }

    direction_single = unit(means["11"] - 0.5 * (means["10"] + means["01"]))
    train_single = train_x @ direction_single
    train_labels = (train_cells == "11").astype(np.int8)
    train_margin = margin(train_labels, train_single)
    if train_margin > 0:
        single_threshold = 0.5 * (
            np.min(train_single[train_labels == 1])
            + np.max(train_single[train_labels == 0])
        )
    else:
        single_threshold = best_threshold(train_single, train_labels)
    single_scores = validation_x @ direction_single - single_threshold

    entity_labels = np.asarray(
        [int(cell[0]) for cell in train_cells], dtype=np.int8
    )
    relation_labels = np.asarray(
        [int(cell[1]) for cell in train_cells], dtype=np.int8
    )
    direction_entity = unit(
        0.5 * ((means["11"] + means["10"]) - (means["01"] + means["00"]))
    )
    direction_relation = unit(
        0.5 * ((means["11"] + means["01"]) - (means["10"] + means["00"]))
    )
    entity_threshold = best_threshold(train_x @ direction_entity, entity_labels)
    relation_threshold = best_threshold(
        train_x @ direction_relation, relation_labels
    )
    conjunction_scores = np.minimum(
        validation_x @ direction_entity - entity_threshold,
        validation_x @ direction_relation - relation_threshold,
    )

    def summarize(scores: np.ndarray) -> dict:
        predicted = scores >= 0
        return {
            "auc": auc(validation_labels, scores),
            "margin": margin(validation_labels, scores),
            "correct": int(np.sum(predicted == validation_labels)),
        }

    return {
        "layer": layer,
        "single": summarize(single_scores),
        "conjunction": summarize(conjunction_scores),
    }


def original_gate_metrics(captures: list[dict]) -> dict:
    cells = np.asarray([capture["cell"] for capture in captures])
    primary = np.isin(cells, ["11", "10", "01"])
    labels = (cells[primary] == "11").astype(np.int8)
    cosine = []
    contribution = []
    for capture in captures:
        events = capture["patch_activations"]
        if len(events) != 1:
            raise ValueError(f"{capture['name']}: expected one patch event, got {len(events)}")
        cosine.append(events[0]["gate_cosine"])
        contribution.append(events[0]["contribution_norm"])
    cosine = np.asarray(cosine, dtype=np.float64)[primary]
    contribution = np.asarray(contribution, dtype=np.float64)[primary]
    return {
        "auc_cosine": auc(labels, cosine),
        "margin_cosine": margin(labels, cosine),
        "auc_contribution": auc(labels, contribution),
        "margin_contribution": margin(labels, contribution),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-output", type=Path)
    parser.add_argument("--conjunction-detector-output", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    captures = json.loads(args.captures.read_text())
    if sorted(capture["cell"] for capture in captures) != sorted(
        [cell for cell in ("11", "10", "01", "00") for _ in range(3)]
    ):
        raise ValueError("expected exactly three captures in each 2x2 cell")
    layers = sorted(
        set.intersection(
            *[
                {item[0] for item in capture["residuals"]}
                for capture in captures
            ]
        )
    )
    results = [
        evaluate_layer(captures, layer, args.random_trials, args.seed)
        for layer in layers
    ]
    best_single = max(
        results,
        key=lambda item: (
            item["auc_single_all_negatives"],
            item["margin_single_all_negatives"],
        ),
    )
    frozen = freeze_single_detector(captures, best_single["layer"])
    report = {
        "n_captures": len(captures),
        "layers": layers,
        "original_gate": original_gate_metrics(captures),
        "best_conjunction": max(results, key=lambda item: item["auc_conjunction"]),
        "best_single": best_single,
        "frozen_single_detector": {
            key: value
            for key, value in frozen.items()
            if key not in ("center", "direction")
        },
        "layer_31": next((item for item in results if item["layer"] == 31), None),
        "all_layer_metrics": [
            {key: value for key, value in item.items() if key != "scores"}
            for item in results
        ],
    }
    if args.validation:
        validation = json.loads(args.validation.read_text())
        validation_metrics = [
            fixed_validation_metrics(captures, validation, layer)
            for layer in layers
        ]
        report["opened_validation_exploration"] = {
            "warning": (
                "Post-hoc only: validation has been opened. Any selected detector "
                "requires a new untouched holdout."
            ),
            "best_single": max(
                validation_metrics,
                key=lambda item: (
                    item["single"]["correct"],
                    item["single"]["margin"],
                ),
            ),
            "best_conjunction": max(
                validation_metrics,
                key=lambda item: (
                    item["conjunction"]["correct"],
                    item["conjunction"]["margin"],
                ),
            ),
            "all_layer_metrics": validation_metrics,
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    if args.detector_output:
        args.detector_output.write_text(json.dumps(frozen) + "\n")
    if args.conjunction_detector_output:
        conjunction = freeze_conjunction_detector(captures, layer=6)
        args.conjunction_detector_output.write_text(json.dumps(conjunction) + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
