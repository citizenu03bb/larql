#!/usr/bin/env python3
"""Build a deterministic one-fact union for one-load development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonical(fact: dict) -> str:
    relation = fact["relation"].replace("-", " ").replace("_", " ")
    return f"The {relation} of {fact['entity']} is"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.suites) < 2:
        raise ValueError("at least two suites are required")

    loaded = []
    for path in args.suites:
        suite = json.loads(path.read_text())
        if len(suite.get("facts", [])) != 1:
            raise ValueError(f"{path} must contain exactly one fact")
        loaded.append((path, suite, suite["facts"][0]))
    first = loaded[0][2]
    identity = (first["entity"], first["target"], first.get("expected_tokens"))
    for path, _, fact in loaded[1:]:
        if (fact["entity"], fact["target"], fact.get("expected_tokens")) != identity:
            raise ValueError(f"{path} does not share entity, target, and token count")

    positives = []
    controls = []
    sources = []
    for path, suite, fact in loaded:
        label = path.stem.removeprefix("multitoken_replication_alder_")
        suite_positives = [canonical(fact), *fact.get("paraphrases", [])]
        if len(set(suite_positives)) != len(suite_positives):
            raise ValueError(f"{path} repeats a positive prompt")
        positives.extend(suite_positives)
        for control in suite.get("controls", []):
            copied = dict(control)
            copied["name"] = f"{label}::{control['name']}"
            controls.append(copied)
        sources.append({"path": str(path), "positive_count": len(suite_positives), "control_count": len(suite.get("controls", []))})
    if len(set(positives)) != len(positives):
        raise ValueError("source suites repeat a positive prompt")
    if len({item["prompt"] for item in controls}) != len(controls):
        raise ValueError("source suites repeat a control prompt")

    combined = {
        "generated_by": "combine_contextual_suites.py",
        "sources": sources,
        "facts": [
            {
                "entity": first["entity"],
                "relation": first["relation"],
                "target": first["target"],
                "expected_tokens": first.get("expected_tokens"),
                "paraphrases": positives[1:],
            }
        ],
        "controls": controls,
    }
    args.output.write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "positives": len(positives), "controls": len(controls)}, indent=2))


if __name__ == "__main__":
    main()
