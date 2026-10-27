#!/usr/bin/env python3
"""Independent model-free B2 v2 control replay; no harness judge imports.

Specification: design sections 3/10/12/13 and the frozen v2 protocol at
2ee2ee4f. Expected headline results were supplied in Pat's brief before
this script was written: this is not claimed as a blind freeze.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[2]
ORDERS = {"subject_before_relation", "subject_after_relation"}
MANDATORY = ("C0", "C1", "C2", "C3", "C4", "C6", "C7")
REVISION = "2ee2ee4f"
INTERPRETATIONS = [
    "Only this fixed 15-case public dev inventory is read. No model loads or blind-directory access.",
    "Native qualification requires an uncapped EOS answer. Efficacy uses old readings; attribute, answer-only and off-target use their declared expect readings. Propagation native qualification is readability only, since expect names the new consequence.",
    "Surface matching is case-sensitive with ASCII-alphanumeric boundaries; swap markers are case-insensitive substrings. The v2 specs supply no ID-based readings.",
    "As in the frozen v1 independent checker, an explicit rejected/old value or swap marker is a counterexample even in a truncated edited answer. A cap hit otherwise remains unreadable, never PASS; completed answers missing the accepted value FAIL. Any discrepancy from the harness is retained.",
    "C1 needs the spec minimum unseen qualified phrasings, covering both orders; a seen phrasing can fail but cannot supply positive coverage.",
    "C2 requires qualified attribute preservation in both orders. A failure takes precedence over missing coverage; unreadability is NOT_EVALUATED.",
    "C3 combines efficacy and attribute preservation in both orders with a qualified answer-only question preserving the declared subject and no reject. Propagation is excluded from C3.",
    "C4 uses raw recorded native/edit/random boundary margins, not recomputed logits. Full diagnostic logits and the actual random adapter arrays are not archived, so their numerical/model provenance is not independently reproduced here.",
    "C6 with declared values tests those readings, not whole-answer token identity. C7 checks readable identical unseen efficacy IDs, unchanged artifact hashes and distinct reload services.",
    "C5 has no donor in these public specs and is NOT_EVALUATED. C8 uses qualified free propagation answers and is non-gating.",
    "The gate counts only C1 PASS and C2 PASS, out of the original five facts; NOT_EVALUATED never counts. A hypothetical upper count treating C2 unknown as preserved is a sensitivity bound, not a success count or probability.",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def has(g, values):
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", g["text"]) for v in values)


def swap(g, spec):
    return any(v.lower() in g["text"].lower() for v in spec.get("swap_markers", []))


def readable(g):
    return bool(g and g["eos"] and not g["capped"])


def generation_status(g, accept, reject, spec):
    if not g:
        return "NOT_EVALUATED"
    if has(g, reject) or swap(g, spec):
        return "FAIL"
    if not readable(g):
        return "NOT_EVALUATED"
    return "PASS" if not accept or has(g, accept) else "FAIL"


def aggregate(rows, enough=True):
    statuses = [r["status"] for r in rows if r["qualified"]]
    if "FAIL" in statuses:
        return "FAIL"
    if statuses and enough and all(s == "PASS" for s in statuses):
        return "PASS"
    return "NOT_EVALUATED"


def replay(spec, evidence, backend, plan):
    bykey = {}
    for g in evidence["generations"]:
        k = (g["question_id"], g["condition"])
        require(k not in bykey, f"Duplicate generation {k}")
        bykey[k] = g
    rows = []
    for q in spec["questions"]:
        require(not any(k in q.get("expect", {}) for k in ("accept_ids", "reject_ids")), "Unexpected token-ID reading")
        native, edited = bykey.get((q["id"], "native")), bykey.get((q["id"], "edit"))
        reading = spec["old"] if q["control"] == "efficacy" else q.get("expect", {})
        qualified = readable(native)
        if qualified and q["control"] != "propagation":
            qualified = (not reading.get("accept") or has(native, reading["accept"])) and not has(native, reading.get("reject", [])) and not swap(native, spec)
        target = spec["new"] if q["control"] == "efficacy" else q.get("expect", {})
        reject = target.get("reject", []) + (spec["old"].get("accept", []) if q["control"] == "efficacy" else [])
        status = generation_status(edited, target.get("accept", []), reject, spec) if qualified else "EXCLUDED"
        if qualified and q["control"] == "off_target" and not q.get("expect"):
            status = "NOT_EVALUATED" if not readable(edited) else ("PASS" if edited["ids"] == native["ids"] else "FAIL")
        rows.append(dict(id=q["id"], control=q["control"], order=q["order"], seen=q.get("seen", False),
                         qualified=bool(qualified), status=status, native=native, edit=edited))
    eff = [r for r in rows if r["control"] == "efficacy"]
    unseen = [r for r in eff if not r["seen"] and r["qualified"]]
    attr = [r for r in rows if r["control"] == "attribute"]
    efficacy_orders = {r["order"] for r in unseen}
    attribute_orders = {r["order"] for r in attr if r["qualified"]}
    efficacy_enough = len(unseen) >= spec.get("min_efficacy", 2) and ORDERS <= efficacy_orders
    controls = {}
    form = backend["form"]
    expected_form = dict(answer_time_trigger=False, donor_pass=False, uses_native_proposal=False,
                         substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=True)
    targets = backend["params"]["targets"]
    require(backend["kind"] == "low_rank" and len(targets) == 1, "Not the fixed rank-one backend")
    require(all(form[k] == v for k, v in expected_form.items()), "Unexpected backend form")
    controls["C0"] = "PASS" if max(t["layer"] for t in targets) <= evidence["kv_write_deadline"] else "FAIL"
    # Seen successes cannot support C1; their failures still refuse it.
    controls["C1"] = "FAIL" if any(r["qualified"] and r["status"] == "FAIL" for r in eff) else aggregate(unseen, efficacy_enough)
    controls["C2"] = aggregate(attr, ORDERS <= attribute_orders)
    answer = [r for r in rows if r["control"] == "answer_only"]
    answer_status = aggregate(answer)
    trio = [controls["C1"], controls["C2"], answer_status]
    controls["C3"] = "FAIL" if "FAIL" in trio else ("PASS" if all(s == "PASS" for s in trio) else "NOT_EVALUATED")
    diagnostics = []
    boundary = {}
    for b in evidence["boundaries"]:
        require(not b["partial"] and not b["variant"], "Unexpected partial/variant boundary in v2")
        k = (b["question_id"], b["condition"])
        require(k not in boundary, f"Duplicate boundary {k}")
        boundary[k] = b
    for r in unseen:
        g = bykey.get((r["id"], "random"))
        triples = [boundary.get((r["id"], c)) for c in ("native", "edit", "random")]
        status, gains = "NOT_EVALUATED", None
        if g and has(g, spec["new"]["accept"]):
            status = "FAIL"
        if all(b is not None and b["margin"] is not None for b in triples):
            n, e, rand = triples
            require(n["native_id"] == e["native_id"] == rand["native_id"] and n["target_id"] == e["target_id"] == rand["target_id"], "Boundary target identity mismatch")
            gains = dict(edit=e["margin"] - n["margin"], random=rand["margin"] - n["margin"])
            greater = gains["edit"] > 2 * max(gains["random"], 0)
            if not greater:
                status = "FAIL"
            elif status != "FAIL" and readable(g):
                status = "PASS"
        diagnostics.append(dict(id=r["id"], qualified=True, status=status, gains=gains))
    controls["C4"] = aggregate(diagnostics, efficacy_enough)
    require(not spec.get("donor"), "Donor not expected in fixed public dev specs")
    controls["C5"] = "NOT_EVALUATED"
    controls["C6"] = aggregate([r for r in rows if r["control"] == "off_target"])
    reload_rows = []
    reload_services = {s["index"] for s in plan["services"] if s["phase"] == "reload"}
    edit_services = {s["index"] for s in plan["services"] if s["phase"] == "edit"}
    require(reload_services and not reload_services & edit_services, "Reload process not distinct")
    for r in unseen:
        g = bykey.get((r["id"], "reloaded"))
        status = "NOT_EVALUATED"
        if readable(r["edit"]) and readable(g):
            status = "PASS" if r["edit"]["ids"] == g["ids"] else "FAIL"
        reload_rows.append(dict(id=r["id"], qualified=True, status=status))
    controls["C7"] = aggregate(reload_rows, efficacy_enough)
    if evidence["artifact_sha256"].get("edit") != evidence["artifact_sha256"].get("reloaded"):
        controls["C7"] = "FAIL"
    controls["C8"] = aggregate([r for r in rows if r["control"] == "propagation"])
    verdict = "CERTIFIED" if all(controls[k] == "PASS" for k in MANDATORY) else "REFUSED"
    return dict(controls=controls, verdict=verdict, questions=rows, random_diagnostics=diagnostics,
                feasibility=dict(efficacy_count=len(unseen), efficacy_orders=sorted(efficacy_orders),
                                 attribute_orders=sorted(attribute_orders), feasible=efficacy_enough and ORDERS <= attribute_orders))


def verify_inputs(root, case, entry):
    run = root / case / "run"
    for name, expected in (("verdict.json", entry["verdict_sha256"]), ("evidence.json", entry["evidence_sha256"])):
        require(sha(run / name) == expected, f"SHA mismatch {case} {name}")
    plan = read(run / "plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    require(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == plan["plan_sha256"] == entry["plan_sha256"], f"Plan SHA mismatch {case}")
    fact = case.rsplit("-L", 1)[0]
    archived_spec = subprocess.check_output(["git", "show", f"{REVISION}:tools/edit_cert/facts/phase-b-dev-v2/{fact}.json"], cwd=REPO)
    require((run / "spec.json").read_bytes() == archived_spec, f"Spec differs from frozen v2 commit {case}")
    require(hashlib.sha256(archived_spec).hexdigest() == plan["spec_sha256"], f"Spec hash mismatch {case}")
    require(len(entry["records_sha256"]) == len(plan["services"]), f"Missing service {case}")
    generations, boundaries, forwards = [], [], []
    for service, digest in zip(plan["services"], entry["records_sha256"]):
        p = run / "services" / str(service["index"]) / "records.json"
        require(sha(p) == digest, f"Record hash mismatch {p}")
        r = read(p)
        require(r["plan_sha256"] == plan["plan_sha256"] and r["service"] == service["index"], f"Record identity mismatch {p}")
        require(r["phase"] == service["phase"] and 0 <= r["forwards"] <= service["forwards"] <= 400, f"Phase/cap mismatch {p}")
        require(not r["synthetic"] and r["instrument"] == "nf4-e2b", f"Wrong instrument {p}")
        for g in r["generations"]:
            require(any(i["kind"] == "generate" and i["question_id"] == g["question_id"] and i["condition"] == g["condition"] for i in service["items"]), f"Unplanned generation {p}")
        generations.extend(r["generations"])
        boundaries.extend(r["boundaries"])
        forwards.append(r["forwards"])
    evidence = read(run / "evidence.json")
    require(generations == evidence["generations"] and boundaries == evidence["boundaries"], f"Aggregated evidence differs {case}")
    require(forwards == entry["forwards"] and sum(forwards) == entry["forwards_total"], f"Forward mismatch {case}")
    require(evidence["kv_write_deadline"] == 13 and not evidence["synthetic"], f"Instrument deadline mismatch {case}")
    return read(run / "spec.json"), evidence, read(run / "edit-source/backend.json"), plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw = (a.root / "index.json").read_bytes()
    index = json.loads(raw)
    facts = read(REPO / "docs/research-data/phase-b0-preflight/budget.json")["facts"]
    expected = {f"{fact}-L{layer}" for fact in facts for layer in (4, 8, 12)}
    require(index["complete"] and set(index["cases"]) == expected, "Incomplete/unexpected fixed inventory")
    output = dict(author="Codex", model_forwards=0, harness_judge_imports=False, claimed_blind_freeze=False,
                  source_revision=REVISION, script_sha256=sha(__file__), index_sha256=hashlib.sha256(raw).hexdigest(),
                  interpretations=INTERPRETATIONS, cases={}, differences=[], feasibility={})
    scores = {str(l): 0 for l in (4, 8, 12)}
    unknown_upper = dict(scores)
    for case, entry in sorted(index["cases"].items()):
        require(entry["state"] == "certified" and entry["algebra_ok"], f"Incomplete certification/algebra {case}")
        spec, evidence, backend, plan = verify_inputs(a.root, case, entry)
        own = replay(spec, evidence, backend, plan)
        peer = read(a.root / case / "run/verdict.json")
        statuses = {c["id"]: c["status"] for c in peer["controls"]}
        require(statuses == entry["statuses"] and peer["verdict"] == entry["verdict"], f"Index/verdict mismatch {case}")
        own["reported_controls"], own["reported_verdict"] = statuses, peer["verdict"]
        own["forwards"] = entry["forwards_total"]
        own["input_sha256"] = {"plan": sha(a.root / case / "run/plan.json"), "spec": sha(a.root / case / "run/spec.json"),
                               "evidence": entry["evidence_sha256"], "verdict": entry["verdict_sha256"], "records": entry["records_sha256"]}
        for c in own["controls"]:
            if own["controls"][c] != statuses[c]:
                output["differences"].append(dict(case=case, control=c, codex=own["controls"][c], claude=statuses[c]))
        if own["verdict"] != peer["verdict"]:
            output["differences"].append(dict(case=case, control="verdict", codex=own["verdict"], claude=peer["verdict"]))
        qualified = {r["id"]: r["qualified"] for r in own["questions"]}
        peer_qualification = {k: v["level"] != "excluded" for k, v in peer["qualification"].items()}
        require(set(qualified) == set(peer_qualification), f"Qualification membership mismatch {case}")
        for k, v in qualified.items():
            if v != peer_qualification[k]:
                output["differences"].append(dict(case=case, control="qualification", question=k, codex=v, claude=peer_qualification[k]))
        layer = case.rsplit("-L", 1)[1]
        scores[layer] += int(own["controls"]["C1"] == own["controls"]["C2"] == "PASS")
        unknown_upper[layer] += int(own["controls"]["C1"] == "PASS" and own["controls"]["C2"] in ("PASS", "NOT_EVALUATED"))
        if case.endswith("-L4"):
            fact = spec["fact_id"]
            output["feasibility"][fact] = own["feasibility"]
            for k, v in own["feasibility"].items():
                require(v == index["feasibility"]["facts"][fact][k], f"Native feasibility discrepancy {fact} {k}")
        output["cases"][case] = own
    output["feasible_facts"] = sum(v["feasible"] for v in output["feasibility"].values())
    require(output["feasible_facts"] == index["feasibility"]["feasible"], "Feasibility count discrepancy")
    output["scores"], output["hypothetical_C2_unknown_upper_counts"] = scores, unknown_upper
    output["gate"] = "STOP_BELOW_2_OF_5" if max(scores.values()) < 2 else "THRESHOLD_MET"
    output["total_certification_forwards"] = sum(c["forwards"] for c in output["cases"].values())
    with a.out.open("x") as f:
        f.write(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: output[k] for k in ("feasible_facts", "scores", "hypothetical_C2_unknown_upper_counts", "gate", "total_certification_forwards", "differences")}))


if __name__ == "__main__":
    main()
