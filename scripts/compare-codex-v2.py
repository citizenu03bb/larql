#!/usr/bin/env python3
"""Closed-map comparison v2: frozen Codex JSON and six emitted harness reports.

No runner/checker/harness imports, raw experiment reads, inference or network.
The 40 original mappings are literal; the only additions are Pat's five
authorized mappings. Unknown, duplicate or missing quantities abort. Existing
outputs are never overwritten. Metadata differs from v1; baseline comparison
payloads (cases/summary/report hashes) retain the original representation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


FREEZE_COMMIT = "5b85fba8f6a147aeae88b30ba813c8dba3081b9c"
FREEZE_SHA256 = "80e61a9034488090d70e0f15b647830513f72b7eeb00388c4a8492d95a2b8262"
ORIGINAL_GENERATOR_SHA256 = "2593968f3866fbf5001446f5879ad22310c8cbca24f1eb891c87d5214e3adeb6"
BASELINE_COMMIT = "f6855066ff9f7117da26a06bf0b7454ccfcdccec"
AMENDED_COMMIT = "620a50b081ec6ad723121fd2591507d610def5e5"

ORIGINAL_MAPPINGS = {
    "R1": {
        "symbol baseline margin": "numbers/symbol_baseline_margin",
        "symbol residual margin": "numbers/symbol_residual_margin",
        "symbol residual argmax": "numbers/symbol_residual_argmax",
        "symbol residual increment": "numbers/symbol_residual_increment",
        "symbol random increment": "numbers/symbol_random_increment",
        "number baseline margin": "numbers/number_baseline_margin",
        "number residual margin": "numbers/number_residual_margin",
        "number residual argmax": "numbers/number_residual_argmax",
        "number residual increment": "numbers/number_residual_increment",
        "number random increment": "numbers/number_random_increment",
        "lithium baseline argmax": "observations/lithium/native_top1",
        "lithium residual argmax": "numbers/lithium_residual_argmax",
        "carbon baseline argmax": "observations/carbon/native_top1",
        "carbon residual argmax": "numbers/carbon_residual_argmax",
    },
    "R2": {
        "selected gain g*": "form/gain",
        "symbol g1.0 margin": "numbers/symbol_g1.0_margin",
        "symbol g1.0 argmax": "numbers/symbol_g1.0_argmax",
        "symbol row5 argmax": "numbers/symbol_row5_argmax",
        "symbol row5 increment": "numbers/symbol_row5_increment",
        "symbol row30 argmax": "numbers/symbol_row30_argmax",
        "symbol row30 increment": "numbers/symbol_row30_increment",
        "number g1.0 margin": "numbers/number_g1.0_margin",
        "number g1.0 argmax": "numbers/number_g1.0_argmax",
        "number row5 argmax": "numbers/number_row5_argmax",
        "number row5 increment": "numbers/number_row5_increment",
        "number row30 argmax": "numbers/number_row30_argmax",
        "number row30 increment": "numbers/number_row30_increment",
    },
    "R3": {
        "symbol random increment": "numbers/symbol_random_increment",
        "number random increment": "numbers/number_random_increment",
        "random float16 norm within 0.1% of the edit norm (protocol rule)": "norm_check/within_tolerance",
    },
    "R4": {
        "symbol donor argmax": "numbers/symbol_donor_argmax",
        "number donor argmax": "numbers/number_donor_argmax",
    },
    "R5": {
        "NUM native margin": "numbers/number_baseline_margin",
        "NUM edited margin": "numbers/number_edit_margin",
        "NUM random 20261025 increment": "numbers/number_random_20261025_increment",
        "NUM random 20261026 increment": "numbers/number_random_20261026_increment",
        "MASS native margin": "numbers/mass_baseline_margin",
        "MASS edited margin": "numbers/mass_edit_margin",
        'mass: rank of "4" native': "numbers/mass_native_four_rank",
        'mass: rank of "4" edited': "numbers/mass_edit_four_rank",
    },
}
ADDED_MAPPINGS = {
    "R1": {"symbol donor argmax": "donor_ceiling_observations/symbol/top1"},
    "R3": {
        "lithium baseline argmax": "observations/lithium/native_top1",
        "lithium random argmax": "numbers/lithium_random_argmax",
        "carbon baseline argmax": "observations/carbon/native_top1",
        "carbon random argmax": "numbers/carbon_random_argmax",
    },
}
FORMS = {
    "response_time_trigger": "answer_time_trigger", "donor_pass": "donor_pass",
    "native_proposal": "uses_native_proposal", "subject_substitution": "substitutes_subject_tokens",
    "unique_stored_artifact": "single_artifact", "applied_each_forward": "applied_every_forward",
    "layer_written": "max_write_layer",
}
QUESTIONS = {"symbol": "SYM", "number": "NUM", "mass": "MASS", "lithium": "LI", "carbon": "C"}
CATEGORIES = ("control_comparisons", "number_comparisons", "form_comparisons",
              "qualification_comparisons", "mention_group_comparisons")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), sha(data)


def lookup(case, location):
    parts = location.split("/")
    if parts[0] == "observations":
        # Exact selector named in the table; no fuzzy/prefix/recursive search.
        matches = [o for o in case["observations"] if o["question"] == parts[1]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one frozen observation: {location}")
        return matches[0][parts[2]]
    value = case
    for part in parts:
        value = value[part]
    return value


def mappings(amended):
    result = {cid: dict(rows) for cid, rows in ORIGINAL_MAPPINGS.items()}
    if sum(map(len, result.values())) != 40 or sum(map(len, ADDED_MAPPINGS.values())) != 5:
        raise ValueError("Invalid mapping manifest size")
    for cid, additions in ADDED_MAPPINGS.items():
        if set(result[cid]) & set(additions):
            raise ValueError("An added mapping replaces an original mapping")
        if amended:
            result[cid].update(additions)
    return result


def compare(freeze_path, reports, harness_commit):
    if harness_commit not in (BASELINE_COMMIT, AMENDED_COMMIT):
        raise ValueError("Unsupported harness commit")
    amended = harness_commit == AMENDED_COMMIT
    x, digest = read_json(freeze_path)
    if digest != FREEZE_SHA256:
        raise ValueError("Frozen JSON SHA-256 differs from 5b85fba8")
    manifest = mappings(amended)
    r, report_digest = read_json(reports / "replay-report.json")
    if set(x["cases"]) != set(manifest) or set(r["cases"]) != set(manifest):
        raise ValueError("Expected exactly R1 through R5")
    result = {
        "author": "Codex", "comparison_performed_after_freeze": True,
        "comparison_completed": True, "freeze_commit": FREEZE_COMMIT,
        "harness_commit": harness_commit, "model_passes": 0,
        "harness_exit_criterion_met": r["exit_criterion_met"],
        "comparison_generator": {"version": 2, "filename": Path(__file__).name,
                                 "sha256": sha(Path(__file__).read_bytes()),
                                 "original_sha256": ORIGINAL_GENERATOR_SHA256},
        "mapping_manifest": {"original": ORIGINAL_MAPPINGS, "added": ADDED_MAPPINGS,
                             "active_count": sum(map(len, manifest.values()))},
        "frozen_json_sha256": digest, "reports_directory": reports.name,
        "harness_report_sha256": {"replay-report.json": report_digest}, "cases": {},
    }
    for cid, c in x["cases"].items():
        h, digest = read_json(reports / f"{cid}-verdict.json")
        result["harness_report_sha256"][f"{cid}-verdict.json"] = digest
        hc = {z["id"]: z for z in h["controls"]}
        if len(hc) != len(h["controls"]) or set(hc) != set(c["controls"]):
            raise ValueError(f"Unexpected or duplicate control IDs in {cid}")
        quantities = [v["quantity"] for v in h["replay"]["reproduced"]]
        if len(set(quantities)) != len(quantities):
            raise ValueError(f"Duplicate replay quantity in {cid}")
        unknown = set(quantities) - set(manifest[cid])
        missing = set(manifest[cid]) - set(quantities)
        if unknown or missing:
            raise ValueError(f"Unmapped or missing quantities in {cid}: unknown={sorted(unknown)}, missing={sorted(missing)}")
        o = {
            "verdict_comparison": {"codex": c["verdict"], "harness_case_report": h["verdict"],
                                   "harness_replay_report": r["cases"][cid]["verdict"],
                                   "equal": c["verdict"] == h["verdict"] == r["cases"][cid]["verdict"]},
            "control_comparisons": [], "number_comparisons": [], "form_comparisons": [],
            "qualification_comparisons": [], "mention_group_comparisons": [], "coverage_notes": [],
        }
        for key, value in c["controls"].items():
            o["control_comparisons"].append({"control": key, "codex": value,
                                              "harness": hc[key]["status"], "equal": value == hc[key]["status"]})
        for key, alias in FORMS.items():
            value, observed = c["form"][key], hc["C0"]["data"]["form"].get(alias)
            o["form_comparisons"].append({"property": key, "harness_property": alias,
                                           "codex": value, "harness": observed, "equal": value == observed})
        for q in sorted(set(h["qualification"]) | {QUESTIONS[z["question"]] for z in c["observations"]}):
            obs = [z for z in c["observations"] if QUESTIONS[z["question"]] == q]
            value = "boundary" if any(z["qualified"] for z in obs) else "excluded"
            observed = h["qualification"].get(q, {}).get("level", "excluded")
            o["qualification_comparisons"].append({"question": q, "codex": value,
                                                    "harness": observed, "equal": value == observed})
        for group, order in (("before", "subject_before_relation"), ("after", "subject_after_relation")):
            value = c["mention_groups"][group]
            observed = "FAIL" if any(z["order"] == order and z["result"] == "fail"
                                     for z in hc["C3"]["outcomes"]) else "NOT_EVALUATED"
            o["mention_group_comparisons"].append({"group": group, "codex": value,
                                                    "harness": observed, "equal": value == observed})
        used = set()
        for v in h["replay"]["reproduced"]:
            q = v["quantity"]
            where = manifest[cid][q]
            value = lookup(c, where)
            if where.split("/")[0] == "numbers":
                used.add(where.split("/")[1])
            o["number_comparisons"].append({
                "harness_quantity": q, "codex_location": where, "codex": value,
                "harness_recomputed": v["recomputed"], "harness_archived_reference": v["archived"],
                "equal": value == v["recomputed"], "equals_harness_reference": value == v["archived"],
            })
        o["codex_numbers_without_harness_counterpart"] = {k: v for k, v in c["numbers"].items() if k not in used}
        if not amended and cid == "R1":
            o["coverage_notes"].append('Codex retains a symbol donor-ceiling observation (top1 He) and separately labels number donor as not applicable to C5. Harness C5 says no donor evidence; both statuses remain NOT_EVALUATED. No numeric mismatch is implied.')
        if not amended and cid == "R3":
            o["coverage_notes"].append('Codex includes the archived LI/C random frontiers as R3 off-target evidence. Harness excludes both questions as no native evidence. Both C6 statuses remain NOT_EVALUATED because no off-target flip occurs.')
        if cid == "R4":
            o["coverage_notes"].append('Codex classified scheduling from the observed donor replay at the native answer boundary. Harness classifies the subject-substitution rule as static. Three C0 form flags differ; C0 status and verdict agree. This is left for Pat to adjudicate.')
        result["cases"][cid] = o
        if amended and cid == "R1":
            value = c["donor_ceiling_observations"]["symbol"]["top1"]
            observed = hc["C5"]["data"]["observed_on_seen"]["SYM"]
            result["seen_donor_observation"] = {"codex": value, "harness": observed,
                                                 "equal": observed == [value], "C5_status": hc["C5"]["status"]}
    summary = {"verdicts_compared": len(result["cases"]),
               "verdict_mismatches": sum(not c["verdict_comparison"]["equal"] for c in result["cases"].values())}
    for category in CATEGORIES:
        items = [z for c in result["cases"].values() for z in c[category]]
        summary[category] = {"compared": len(items), "mismatches": sum(not z["equal"] for z in items)}
    summary["numeric_reference_mismatches"] = sum(not n["equals_harness_reference"] for c in result["cases"].values() for n in c["number_comparisons"])
    summary["codex_numeric_items_without_harness_counterpart"] = sum(len(c["codex_numbers_without_harness_counterpart"]) for c in result["cases"].values())
    result["summary"] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--harness-commit", choices=(BASELINE_COMMIT, AMENDED_COMMIT), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.freeze, args.reports, args.harness_commit)
        with args.out.open("x") as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"comparison_completed": False, "error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
