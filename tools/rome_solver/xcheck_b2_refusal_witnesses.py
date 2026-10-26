#!/usr/bin/env python3
"""Sufficient independent witnesses for REFUSED, not all-control replay.

Written after the first peer comparisons. Uses the independently frozen
C1/C2 readings and explicit C3 order-coverage rule, never harness judges.
One mandatory control that cannot PASS proves the whole verdict REFUSED.
No claim is made to independently validate any positive certification.
"""
import argparse
import hashlib
import json
from pathlib import Path

ORDERS = {"subject_before_relation", "subject_after_relation"}


def witness(case):
    for control in ("C1", "C2"):
        own = case["controls"][control]
        if own != "PASS" and own == case["reported_controls"][control]:
            return dict(control=control, cannot_pass=True, independently_recomputed=own,
                        reason="Frozen independent control reading agrees with reported non-PASS")
    missing = ORDERS - {q["order"] for q in case["questions"]
                        if q["control"] == "attribute" and q["qualified"]}
    if missing:
        return dict(control="C3", cannot_pass=True, independently_recomputed="NOT_PASS",
                    reason="C3 requires attribute preservation in each order; no native-qualified attribute in: " + ", ".join(sorted(missing)))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw = a.comparison.read_bytes()
    source = json.loads(raw)
    output = dict(author="Codex", model_forwards=0, post_comparison_extension=True,
                  exhaustive_control_status_replay=False, validates_positive_certifications=False,
                  comparison_sha256=hashlib.sha256(raw).hexdigest(), cases={}, differences=[])
    for name, case in source["cases"].items():
        if "reported_controls" not in case:
            output["cases"][name] = dict(state=case["state"], verified=False)
            continue
        path = a.root / name / "run/verdict.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["input_sha256"]["verdict"]:
            raise ValueError(f"Verdict changed after raw comparison: {name}")
        verdict = json.loads(path.read_text())["verdict"]
        proof = witness(case)
        verified = proof is not None and verdict == "REFUSED"
        output["cases"][name] = dict(reported_verdict=verdict, independent_verdict="REFUSED" if proof else "NOT_EVALUATED",
                                     sufficient_witness=proof, verified=verified)
        if proof and verdict != "REFUSED":
            output["differences"].append(dict(case=name, reported=verdict, independent="REFUSED"))
    output["complete"] = source["peer_complete"] and not source["pending"] and len(source["cases"]) == 15
    output["all_15_refusals_independently_supported"] = output["complete"] and all(c["verified"] for c in output["cases"].values())
    output["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open("x") as f:
        f.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(output["cases"]), "verified_refusals": sum(c["verified"] for c in output["cases"].values()),
                      "complete": output["complete"], "verdict_differences": len(output["differences"])}))


if __name__ == "__main__":
    main()
