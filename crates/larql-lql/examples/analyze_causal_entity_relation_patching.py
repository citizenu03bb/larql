#!/usr/bin/env python3
"""Evaluate the frozen causal entity-relation activation-patching run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SITES = (11, 13, 12, 33)
CANDIDATES = (11, 13)
FOLDS = tuple(range(6))
DISCOVERY_FOLDS = (0, 1, 2, 3)
CONFIRMATION_FOLDS = (4, 5)
ARMS = (
    "matched_forward",
    "matched_inverse",
    "positive_self",
    "control_self",
    "entity_absent_forward",
    "entity_absent_inverse",
    "nonpaired_forward",
    "nonpaired_inverse",
)
SELF_TOLERANCE = 1e-4
SPECIFICITY_FLOOR = 0.001


def fail(message: str) -> None:
    raise ValueError(f"analysis-inconclusive: {message}")


def finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{context} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{context} is non-finite")
    return result


def validate_baselines(run: dict) -> None:
    baselines = run.get("baselines")
    if not isinstance(baselines, list) or len(baselines) != 24:
        fail("expected exactly 24 baselines")
    expected = {(fold, cell) for fold in FOLDS for cell in ("11", "10", "01", "00")}
    observed: set[tuple[int, str]] = set()
    for index, row in enumerate(baselines):
        key = (row.get("fold"), row.get("cell"))
        if key in observed:
            fail(f"duplicate baseline {key}")
        observed.add(key)
        if row.get("noop_digest_identical") is not True:
            fail(f"baseline {key} failed hook-disabled digest identity")
        if row.get("full_logits_sha256") != row.get("noop_full_logits_sha256"):
            fail(f"baseline {key} reports inconsistent full-logit digests")
        for field in ("target_logit", "distractor_logit", "gap"):
            finite_number(row.get(field), context=f"baseline {index} {field}")
    if observed != expected:
        fail(f"baseline keys differ: missing={sorted(expected - observed)} extra={sorted(observed - expected)}")


def validate_arms(run: dict) -> dict[tuple[int, int, str], dict]:
    rows = run.get("arms")
    if not isinstance(rows, list) or len(rows) != 192:
        fail("expected exactly 192 patched arms")
    expected = {(site, fold, arm) for site in SITES for fold in FOLDS for arm in ARMS}
    indexed: dict[tuple[int, int, str], dict] = {}
    for index, row in enumerate(rows):
        key = (row.get("site"), row.get("fold"), row.get("arm"))
        if key in indexed:
            fail(f"duplicate patched arm {key}")
        indexed[key] = row
        site, _, _ = key
        position = row.get("recipient_position")
        if row.get("applied") != [[site, position]] or row.get("rejected") != []:
            fail(f"patched arm {key} has invalid hook accounting")
        for field in (
            "target_logit",
            "distractor_logit",
            "gap",
            "delta_target",
            "delta_distractor",
            "delta_gap",
        ):
            finite_number(row.get(field), context=f"arm {index} {field}")
    if set(indexed) != expected:
        fail(f"arm keys differ: missing={sorted(expected - set(indexed))} extra={sorted(set(indexed) - expected)}")
    return indexed


def validate_run(run: dict) -> dict[tuple[int, int, str], dict]:
    if run.get("schema") != "larql.causal_entity_relation_patching.v1":
        fail("unexpected run schema")
    if run.get("sites") != list(SITES):
        fail("site order or membership changed")
    if run.get("target_id") != 107545 or run.get("distractor_id") != 506:
        fail("target or distractor ID changed")
    expected_counts = {
        "baseline_forwards": 24,
        "noop_forwards": 24,
        "patched_forwards": 192,
        "total_forwards": 240,
    }
    for field, expected in expected_counts.items():
        if run.get(field) != expected:
            fail(f"{field} must equal {expected}")
    validate_baselines(run)
    indexed = validate_arms(run)
    for site in SITES:
        for fold in FOLDS:
            for arm in ("positive_self", "control_self"):
                row = indexed[(site, fold, arm)]
                drift = max(abs(float(row[field])) for field in ("delta_gap", "delta_target", "delta_distractor"))
                if drift > SELF_TOLERANCE:
                    fail(f"L{site} fold {fold} {arm} self-patch drift {drift} exceeds {SELF_TOLERANCE}")
    return indexed


def fold_metrics(indexed: dict[tuple[int, int, str], dict], site: int, fold: int) -> dict:
    row = lambda arm: indexed[(site, fold, arm)]
    m_plus = float(row("matched_forward")["delta_gap"])
    m_minus = -float(row("matched_inverse")["delta_gap"])
    t_plus = float(row("matched_forward")["delta_target"])
    t_minus = -float(row("matched_inverse")["delta_target"])
    control_arms = (
        "entity_absent_forward",
        "entity_absent_inverse",
        "nonpaired_forward",
        "nonpaired_inverse",
    )
    max_control_gap = max(abs(float(row(arm)["delta_gap"])) for arm in control_arms)
    max_control_target = max(abs(float(row(arm)["delta_target"])) for arm in control_arms)
    matched_minimum = min(m_plus, m_minus)
    return {
        "site": site,
        "fold": fold,
        "m_plus": m_plus,
        "m_minus": m_minus,
        "t_plus": t_plus,
        "t_minus": t_minus,
        "matched_minimum": matched_minimum,
        "max_absolute_control_gap": max_control_gap,
        "max_absolute_control_target": max_control_target,
        "gap_specificity": matched_minimum - max_control_gap,
        "target_specificity": min(t_plus, t_minus) - max_control_target,
    }


def add_gate(metrics: dict, adjacent: dict) -> dict:
    adjacent_advantage = metrics["matched_minimum"] - adjacent["matched_minimum"]
    metrics["adjacent_l12_matched_minimum"] = adjacent["matched_minimum"]
    metrics["adjacent_advantage"] = adjacent_advantage
    metrics["passes"] = (
        metrics["m_plus"] > 0.0
        and metrics["m_minus"] > 0.0
        and metrics["t_plus"] > 0.0
        and metrics["t_minus"] > 0.0
        and metrics["gap_specificity"] > SPECIFICITY_FLOOR
        and metrics["target_specificity"] > SPECIFICITY_FLOOR
        and adjacent_advantage > SPECIFICITY_FLOOR
    )
    return metrics


def analyze(run: dict) -> dict:
    indexed = validate_run(run)
    raw = {
        (site, fold): fold_metrics(indexed, site, fold)
        for site in SITES
        for fold in FOLDS
    }
    reports = []
    candidate_passes: dict[int, bool] = {}
    for site in SITES:
        folds = []
        for fold in FOLDS:
            metrics = raw[(site, fold)]
            if site in CANDIDATES:
                metrics = add_gate(metrics, raw[(12, fold)])
            else:
                metrics["passes"] = None
            folds.append(metrics)
        discovery_passes = site in CANDIDATES and all(
            folds[fold]["passes"] for fold in DISCOVERY_FOLDS
        )
        if site in CANDIDATES:
            candidate_passes[site] = discovery_passes
        reports.append(
            {
                "site": site,
                "role": (
                    "candidate"
                    if site in CANDIDATES
                    else "adjacent negative control"
                    if site == 12
                    else "late sensitivity control"
                ),
                "discovery_passes": discovery_passes if site in CANDIDATES else None,
                "folds": folds,
            }
        )

    selected = 11 if candidate_passes[11] else 13 if candidate_passes[13] else None
    selected_report = next((report for report in reports if report["site"] == selected), None)
    confirmation_passes = (
        all(selected_report["folds"][fold]["passes"] for fold in CONFIRMATION_FOLDS)
        if selected_report is not None
        else False
    )
    nominated = selected if selected is not None and confirmation_passes else None
    return {
        "version": 1,
        "epistemic_status": (
            "exploratory causal intervention on spent construction groups and an engineered "
            "source patch; not fresh holdout, certification, natural factual recall, or cross-model evidence"
        ),
        "self_tolerance": SELF_TOLERANCE,
        "specificity_floor": SPECIFICITY_FLOOR,
        "discovery_folds": list(DISCOVERY_FOLDS),
        "construction_confirmation_folds": list(CONFIRMATION_FOLDS),
        "candidate_discovery_passes": {str(site): candidate_passes[site] for site in CANDIDATES},
        "selected_discovery_layer": selected,
        "selected_confirmation_passes": confirmation_passes,
        "nominated_layer": nominated,
        "verdict": (
            "exploratory causal mediator nominated"
            if nominated is not None
            else "exploratory causal candidate rejected"
        ),
        "sites": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.run.read_text()))
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
