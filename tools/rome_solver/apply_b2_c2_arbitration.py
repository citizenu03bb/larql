#!/usr/bin/env python3
"""Apply Pat's explicit post-comparison C2 coverage arbitration, model-free.

Never rewrites either the frozen checker, peer verdicts or comparison inputs.
Only a reported C2 PASS lacking qualified native coverage in BOTH orders
becomes NOT_EVALUATED for the development decision. Other differences stay
open and block a positive continuation decision pending arbitration.
"""
import argparse
import hashlib
import json
from pathlib import Path

ORDERS = {"subject_before_relation", "subject_after_relation"}
REPO = Path(__file__).resolve().parents[2]


def decide(source):
    budget = json.loads((REPO / "docs/research-data/phase-b0-preflight/budget.json").read_text())
    expected = {f"{fact}-L{layer}" for fact in budget["facts"] for layer in (4, 8, 12)}
    if not set(source["cases"]) <= expected:
        raise ValueError("Unexpected development case")
    counts = {str(layer): 0 for layer in (4, 8, 12)}
    cases, covered_differences = {}, set()
    for case, data in source["cases"].items():
        if "reported_controls" not in data:
            cases[case] = dict(state=data["state"], counted_pass=False)
            continue
        reported = data["reported_controls"]
        effective = dict(reported)
        orders = {q["order"] for q in data["questions"] if q["control"] == "attribute" and q["qualified"]}
        amendment = None
        if reported["C2"] == "PASS" and not ORDERS <= orders:
            effective["C2"] = "NOT_EVALUATED"
            amendment = "Pat C2 arbitration: both native-qualified subject orders required"
            covered_differences.add((case, "C2", "NOT_EVALUATED", "PASS"))
        passed = all(s == "PASS" for s in effective.values())
        counts[case.rsplit("-L", 1)[1]] += int(passed)
        cases[case] = dict(reported=reported, decision_controls=effective,
                           qualified_attribute_orders=sorted(orders), amendment=amendment, counted_pass=passed)
    unresolved = [d for d in source["differences"]
                  if (d["case"], d["control"], d["codex"], d["claude"]) not in covered_differences]
    complete = source["peer_complete"] and not source["pending"] and set(source["cases"]) == expected
    winner = max((4, 8, 12), key=lambda layer: (counts[str(layer)], -layer))
    if not complete:
        gate = "PENDING"
    elif counts[str(winner)] < 2:
        gate = "STOP_BELOW_2_OF_5"
    elif unresolved:
        gate = "PENDING_OTHER_ARBITRATIONS"
    else:
        gate = "CONTINUE_IDENTITY_ABLATION"
    return dict(author="Codex", amendment="Pat C2 native-qualified coverage in both orders, 2026-10-26",
                original_checker_freeze="1d6af9e8", outputs_rewritten=False, model_forwards=0,
                complete=complete, cases=cases, pass_counts=counts, unresolved_differences=unresolved,
                gate=gate, selected_layer=winner if gate == "CONTINUE_IDENTITY_ABLATION" else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    raw = args.comparison.read_bytes()
    report = decide(json.loads(raw))
    report["comparison_sha256"] = hashlib.sha256(raw).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open("x") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("complete", "pass_counts", "gate", "selected_layer")}))


if __name__ == "__main__":
    main()
