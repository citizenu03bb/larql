#!/usr/bin/env python3
"""Leave-one-suite-out study of deployable multi-position detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_contextual_calibration import auc, best_threshold, unit, unit_rows


REPRESENTATIONS = (
    "last",
    "relative_2",
    "relative_3",
    "relative_4",
    "mean_no_bos",
    "mean_body",
    "mean_last2",
    "last_minus_previous",
    "concat_last2",
    "concat_last3",
)
METHODS = (
    "raw_conjunction",
    "unit_conjunction",
    "unit_factor_prototypes",
    "unit_cell_prototypes",
)
DEPLOYABLE_REPRESENTATIONS = ("mean_no_bos",)
DEPLOYABLE_METHODS = (
    "raw_conjunction",
    "unit_conjunction",
    "unit_factor_prototypes",
)


def positions_at(capture: dict, layer: int) -> np.ndarray:
    by_layer = {item[0]: item[1] for item in capture["position_residuals"]}
    return np.asarray(by_layer[layer], dtype=np.float64)


def represent(positions: np.ndarray, name: str) -> np.ndarray:
    if name == "last":
        return positions[-1]
    if name.startswith("relative_"):
        offset = int(name.removeprefix("relative_"))
        return positions[-offset]
    if name == "mean_no_bos":
        return np.mean(positions[1:], axis=0)
    if name == "mean_body":
        return np.mean(positions[1:-1], axis=0)
    if name == "mean_last2":
        return np.mean(positions[-2:], axis=0)
    if name == "last_minus_previous":
        return positions[-1] - positions[-2]
    if name == "concat_last2":
        return positions[-2:].reshape(-1)
    if name == "concat_last3":
        return positions[-3:].reshape(-1)
    raise ValueError(f"unknown representation: {name}")


def matrix(captures: list[dict], layer: int, representation: str) -> np.ndarray:
    if representation == "last" and all(
        not capture.get("position_residuals") for capture in captures
    ):
        rows = []
        for capture in captures:
            residuals = {int(item[0]): item[1] for item in capture["residuals"]}
            rows.append(residuals[layer])
        return np.asarray(rows, dtype=np.float64)
    return np.asarray(
        [represent(positions_at(capture, layer), representation) for capture in captures],
        dtype=np.float64,
    )


def fit(x: np.ndarray, cells: np.ndarray, method: str) -> dict:
    center = np.mean(x, axis=0)
    transformed = x - center
    if method.startswith("unit_"):
        transformed = unit_rows(transformed)
    means = {
        cell: np.mean(transformed[cells == cell], axis=0)
        for cell in ("11", "10", "01", "00")
    }
    detector: dict = {"method": method, "center": center}
    if method.endswith("conjunction"):
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
        detector.update(
            entity_present=unit(
                np.mean(transformed[[cell[0] == "1" for cell in cells]], axis=0)
            ),
            entity_absent=unit(
                np.mean(transformed[[cell[0] == "0" for cell in cells]], axis=0)
            ),
            relation_present=unit(
                np.mean(transformed[[cell[1] == "1" for cell in cells]], axis=0)
            ),
            relation_absent=unit(
                np.mean(transformed[[cell[1] == "0" for cell in cells]], axis=0)
            ),
        )
    elif method == "unit_cell_prototypes":
        detector["prototypes"] = {cell: unit(mean) for cell, mean in means.items()}
    else:
        raise ValueError(f"unsupported method: {method}")
    return detector


def score(detector: dict, x: np.ndarray) -> np.ndarray:
    transformed = x - detector["center"]
    method = detector["method"]
    if method.startswith("unit_"):
        transformed = unit_rows(transformed)
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
    similarities = {
        cell: transformed @ prototype
        for cell, prototype in detector["prototypes"].items()
    }
    return similarities["11"] - np.maximum.reduce(
        [similarities[cell] for cell in ("10", "01", "00")]
    )


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    return value


def runtime_detector(detector: dict, representation: str) -> dict:
    """Convert an accepted search candidate to the exact runtime ABI."""
    if representation not in DEPLOYABLE_REPRESENTATIONS:
        raise ValueError(f"representation is not deployable: {representation}")
    method = detector["method"]
    if method.endswith("conjunction"):
        entity_direction = detector["entity_direction"]
        relation_direction = detector["relation_direction"]
        entity_threshold = detector["entity_threshold"]
        relation_threshold = detector["relation_threshold"]
    elif method == "unit_factor_prototypes":
        entity_direction = detector["entity_present"] - detector["entity_absent"]
        relation_direction = detector["relation_present"] - detector["relation_absent"]
        entity_threshold = 0.0
        relation_threshold = 0.0
    else:
        raise ValueError(f"method is not deployable: {method}")
    return serializable(
        {
            "center": detector["center"],
            "entity_direction": entity_direction,
            "relation_direction": relation_direction,
            "entity_threshold": entity_threshold,
            "relation_threshold": relation_threshold,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="+", help="NAME=PATH capture files")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-output", type=Path)
    parser.add_argument(
        "--deployable-only",
        action="store_true",
        help="accept only candidates exactly representable by the current runtime ABI",
    )
    args = parser.parse_args()

    suites: dict[str, list[dict]] = {}
    for item in args.suites:
        name, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"suite must be NAME=PATH, got {item!r}")
        suites[name] = json.loads(Path(path).read_text())
    if len(suites) < 3:
        raise ValueError("expected at least three suites")
    all_captures = [capture for captures in suites.values() for capture in captures]
    if any(not capture.get("position_residuals") for capture in all_captures):
        raise ValueError("every capture must contain position_residuals")
    if any(len(capture["prompt_ids"]) < 4 for capture in all_captures):
        raise ValueError("relative/concat representations require at least four tokens")
    layers = sorted(
        set.intersection(
            *[
                {item[0] for item in capture["position_residuals"]}
                for capture in all_captures
            ]
        )
    )
    matrices = {
        name: {
            (layer, representation): matrix(captures, layer, representation)
            for layer in layers
            for representation in REPRESENTATIONS
        }
        for name, captures in suites.items()
    }
    cells_by_suite = {
        name: np.asarray([capture["cell"] for capture in captures])
        for name, captures in suites.items()
    }
    # The JSON lists dominate memory once their numeric matrices have been
    # materialised. Keep only prompt metadata for reporting.
    for capture in all_captures:
        capture["position_residuals"] = []

    candidates = []
    for layer in layers:
        for representation in REPRESENTATIONS:
            for method in METHODS:
                rows = []
                suite_metrics = []
                for heldout_name, heldout in suites.items():
                    train_names = [name for name in suites if name != heldout_name]
                    train_x = np.concatenate(
                        [matrices[name][(layer, representation)] for name in train_names]
                    )
                    train_cells = np.concatenate(
                        [cells_by_suite[name] for name in train_names]
                    )
                    detector = fit(train_x, train_cells, method)
                    scores = score(
                        detector, matrices[heldout_name][(layer, representation)]
                    )
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
                            "score": float(value),
                            "correct": bool(margin >= 0),
                        }
                        for capture, value, margin in zip(heldout, scores, signed)
                    )
                labels = np.asarray(
                    [row["cell"] == "11" for row in rows], dtype=np.int8
                )
                scores = np.asarray([row["score"] for row in rows])
                signed = np.where(labels == 1, scores, -scores)
                candidates.append(
                    {
                        "layer": layer,
                        "representation": representation,
                        "method": method,
                        "correct": int(np.sum(signed >= 0)),
                        "n": len(rows),
                        "decision_margin": float(np.min(signed)),
                        "ranking_margin": float(
                            np.min(scores[labels == 1])
                            - np.max(scores[labels == 0])
                        ),
                        "auc": auc(labels, scores),
                        "runtime_compatible": representation in DEPLOYABLE_REPRESENTATIONS
                        and method in DEPLOYABLE_METHODS,
                        "suite_metrics": suite_metrics,
                        "rows": rows,
                    }
                )

    accepted = [
        candidate
        for candidate in candidates
        if candidate["correct"] == candidate["n"]
        and candidate["decision_margin"] > 0
        and (not args.deployable_only or candidate["runtime_compatible"])
    ]
    ordering = lambda candidate: (
        candidate["correct"],
        min(item["correct"] / item["n"] for item in candidate["suite_metrics"]),
        candidate["decision_margin"],
        candidate["ranking_margin"],
    )
    best = max(candidates, key=ordering)
    last_best = max(
        (candidate for candidate in candidates if candidate["representation"] == "last"),
        key=ordering,
    )
    report = {
        "protocol": "leave-one-entire-suite-out",
        "suite_sizes": {name: len(captures) for name, captures in suites.items()},
        "n_captures": len(all_captures),
        "layers": layers,
        "representations": list(REPRESENTATIONS),
        "methods": list(METHODS),
        "acceptance": (
            f"{len(all_captures)}/{len(all_captures)} decisions, positive worst "
            "signed margin, and runtime-compatible ABI"
            if args.deployable_only
            else f"{len(all_captures)}/{len(all_captures)} decisions and positive worst signed margin"
        ),
        "deployable_only": args.deployable_only,
        "deployable_representations": list(DEPLOYABLE_REPRESENTATIONS),
        "deployable_methods": list(DEPLOYABLE_METHODS),
        "accepted_candidates": len(accepted),
        "last_position_control": {
            key: value for key, value in last_best.items() if key != "rows"
        },
        "best": best,
        "top_candidates": [
            {key: value for key, value in candidate.items() if key != "rows"}
            for candidate in sorted(candidates, key=ordering, reverse=True)[:30]
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    if args.detector_output:
        if not accepted:
            raise ValueError("no candidate met acceptance; refusing to freeze")
        selected = max(accepted, key=ordering)
        selected_x = np.concatenate(
            [
                matrices[name][(selected["layer"], selected["representation"])]
                for name in suites
            ]
        )
        selected_cells = np.concatenate(
            [cells_by_suite[name] for name in suites]
        )
        detector = fit(selected_x, selected_cells, selected["method"])
        frozen = {
            "version": 1,
            "selection_protocol": (
                f"leave-one-entire-suite-out on {len(all_captures)} development captures"
            ),
            "layer": selected["layer"],
            "representation": selected["representation"],
            "selection_metrics": {
                key: value
                for key, value in selected.items()
                if key not in ("rows", "suite_metrics")
            },
            "method": selected["method"],
            **runtime_detector(detector, selected["representation"]),
        }
        args.detector_output.write_text(json.dumps(frozen) + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
