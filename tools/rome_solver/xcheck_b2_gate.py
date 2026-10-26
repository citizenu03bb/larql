#!/usr/bin/env python3
"""Independent, model-free C1/C2 check from raw service generations.

No edit_cert imports or judging-code reads. --compare is deliberately opt-in:
the first output can be frozen before looking at the peer's control results.
This does NOT independently certify C0 or C3--C8.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORDERS = {"subject_before_relation", "subject_after_relation"}
LAYERS = (4, 8, 12)
INTERPRETATIONS = [
    "Native qualification requires EOS, no cap, an accepted native surface, no rejected surface or swap marker.",
    "Surface readings are case-sensitive, bounded by non-ASCII-alphanumeric-or-edge, as declared by the public specs; swap markers are case-insensitive substrings.",
    "C1 requires min_efficacy unseen qualified positive generations in total and at least one in each subject order (design section 3); it does not reinterpret the later test success threshold as two per order.",
    "A seen efficacy phrasing can fail C1 but cannot supply positive coverage.",
    "An explicit old/rejected value or swap marker fails even in a capped edited generation. A capped answer without such counterevidence is unreadable, never PASS. A completed answer missing its accepted value fails.",
    "C2 needs positive qualified attribute coverage in both subject orders; insufficient coverage is NOT_EVALUATED. Any usable counterexample takes precedence.",
    "Primary B2 has full free generations. Resumed boundary diagnostics are not used to rescue, replace, or score them.",
    "Layer selection is the frozen count of REPORTED C1+C2 PASS among the original five facts, ties to the earlier layer. This cross-check reports differences without repairing either side.",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def contains(generation, surfaces):
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])",
                         generation["text"]) is not None for s in surfaces)


def swapped(generation, spec):
    return any(s.lower() in generation["text"].lower() for s in spec.get("swap_markers", []))


def readable(generation):
    return generation is not None and generation["eos"] and not generation["capped"]


def native_ok(generation, reading, spec):
    return bool(readable(generation) and contains(generation, reading.get("accept", []))
                and not contains(generation, reading.get("reject", [])) and not swapped(generation, spec))


def edited_status(generation, reading, reject, spec):
    if generation is None:
        return "NOT_EVALUATED"
    if contains(generation, reject) or swapped(generation, spec):
        return "FAIL"
    if not readable(generation):
        return "NOT_EVALUATED"
    return "PASS" if contains(generation, reading.get("accept", [])) else "FAIL"


def controls(spec, generations):
    by_key = {}
    for generation in generations:
        key = (generation["question_id"], generation["condition"])
        require(key not in by_key, f"Duplicate generation {key}")
        by_key[key] = generation
    questions = []
    for q in spec["questions"]:
        if q["control"] not in ("efficacy", "attribute"):
            continue
        reading = spec["old"] if q["control"] == "efficacy" else q["expect"]
        native = by_key.get((q["id"], "native"))
        edit = by_key.get((q["id"], "edit"))
        qualified = native_ok(native, reading, spec)
        target = spec["new"] if q["control"] == "efficacy" else reading
        reject = (spec["old"].get("accept", []) + target.get("reject", [])
                  if q["control"] == "efficacy" else reading.get("reject", []))
        status = edited_status(edit, target, reject, spec) if qualified else "EXCLUDED"
        questions.append(dict(id=q["id"], control=q["control"], order=q["order"],
                              seen=q.get("seen", False), qualified=qualified, edit_status=status,
                              native=native, edit=edit))
    result = {}
    for name, control in (("C1", "efficacy"), ("C2", "attribute")):
        used = [q for q in questions if q["control"] == control and q["qualified"]]
        positive = [q for q in used if not q["seen"]]
        coverage = {q["order"] for q in positive if q["edit_status"] == "PASS"}
        enough = len(positive) >= (spec.get("min_efficacy", 2) if name == "C1" else 2)
        if any(q["edit_status"] == "FAIL" for q in used):
            status = "FAIL"
        elif enough and ORDERS <= coverage and all(q["edit_status"] == "PASS" for q in positive):
            status = "PASS"
        else:
            status = "NOT_EVALUATED"
        result[name] = status
    return result, questions


def check_case(root, case, entry, expected_spec_sha):
    run = root / case / "run"
    plan = read(run / "plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    require(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            == plan["plan_sha256"] == entry["plan_sha256"], f"Plan hash mismatch: {case}")
    require(sha(run / "spec.json") == expected_spec_sha == plan["spec_sha256"], f"Spec hash mismatch: {case}")
    require(sha(run / "evidence.json") == entry["evidence_sha256"], f"Evidence hash mismatch: {case}")
    require(sha(run / "verdict.json") == entry["verdict_sha256"], f"Verdict hash mismatch: {case}")
    evidence = read(run / "evidence.json")
    generations, boundaries, forwards, records = [], [], [], []
    require(not evidence["synthetic"] and evidence["instrument"] == "nf4-e2b", f"Wrong evidence: {case}")
    require(len(entry["records_sha256"]) == len(plan["services"]), f"Service count mismatch: {case}")
    for service, digest in zip(plan["services"], entry["records_sha256"]):
        path = run / "services" / str(service["index"]) / "records.json"
        require(sha(path) == digest, f"Record hash mismatch: {path}")
        r = read(path)
        require(r["service"] == service["index"] and r["plan_sha256"] == plan["plan_sha256"], f"Record identity mismatch: {path}")
        require(not r["synthetic"] and r["instrument"] == "nf4-e2b", f"Wrong instrument: {path}")
        require(0 <= r["forwards"] <= service["forwards"] <= 400, f"Forward cap mismatch: {path}")
        generations.extend(r["generations"])
        boundaries.extend(r["boundaries"])
        forwards.append(r["forwards"])
        records.append(dict(path=str(path), sha256=digest, forwards=r["forwards"]))
    require(generations == evidence["generations"], f"Evidence generations differ from raw records: {case}")
    require(boundaries == evidence["boundaries"], f"Evidence boundaries differ from raw records: {case}")
    require(forwards == entry["forwards"] and sum(forwards) == entry["forwards_total"], f"Index counters mismatch: {case}")
    values, questions = controls(read(run / "spec.json"), generations)
    return dict(controls=values, questions=questions, records=records,
                input_sha256={"plan": sha(run / "plan.json"), "spec": expected_spec_sha,
                              "evidence": sha(run / "evidence.json"), "verdict": sha(run / "verdict.json")},
                forwards=sum(forwards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", action="store_true")
    a = parser.parse_args()
    index_path = a.root / "index.json"
    raw_index = index_path.read_bytes()  # snapshot once while the peer may be appending
    index = json.loads(raw_index)
    budget = read(REPO / "docs/research-data/phase-b0-preflight/budget.json")
    expected = {f"{fact}-L{layer}": data["spec_sha256"] for fact, data in budget["facts"].items() for layer in LAYERS}
    require(set(index["cases"]) <= set(expected), "Unexpected case in primary index")
    output = dict(author="Codex", model_forwards=0, independent_scope=["C1", "C2"],
                  full_certification=False, interpretations=INTERPRETATIONS,
                  comparator_sha256=sha(__file__), peer_complete=index.get("complete", False),
                  input_index_sha256=hashlib.sha256(raw_index).hexdigest(), cases={}, pending=[], differences=[])
    counts = {str(layer): 0 for layer in LAYERS}
    for case, spec_sha in sorted(expected.items()):
        entry = index["cases"].get(case)
        if entry is None:
            output["pending"].append(case)
            continue
        if entry["state"] != "certified":
            output["cases"][case] = {"state": entry["state"], "counted_pass": False}
            continue
        own = check_case(a.root, case, entry, spec_sha)
        output["cases"][case] = own
        if a.compare:
            verdict = read(a.root / case / "run/verdict.json")
            reported = {c["id"]: c["status"] for c in verdict["controls"] if c["id"] in ("C1", "C2")}
            for name in ("C1", "C2"):
                require(entry[name] == entry["statuses"][name] == reported[name], f"Index/verdict mismatch: {case} {name}")
                if own["controls"][name] != reported[name]:
                    output["differences"].append(dict(case=case, control=name, codex=own["controls"][name], claude=reported[name]))
            own["reported_controls"] = reported
            own["reported_verdict"] = verdict["verdict"]
            counts[case.rsplit("-L", 1)[1]] += int(all(s == "PASS" for s in reported.values()))
    if index.get("complete"):
        require(not output["pending"] and len(index["cases"]) == 15, "Index says complete but is incomplete")
    if a.compare:
        output["reported_pass_counts"] = counts
        complete = index.get("complete", False) and not output["pending"]
        winner = max(LAYERS, key=lambda layer: (counts[str(layer)], -layer))
        output["gate"] = ("CONTINUE_IDENTITY_ABLATION" if counts[str(winner)] >= 2 else "STOP_BELOW_2_OF_5") if complete else "PENDING"
        output["selected_layer"] = winner if complete and counts[str(winner)] >= 2 else None
    with a.out.open("x") as f:
        f.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checked": len(output["cases"]), "pending": len(output["pending"]),
                      "compared": a.compare, "differences": len(output["differences"])}))


if __name__ == "__main__":
    main()
