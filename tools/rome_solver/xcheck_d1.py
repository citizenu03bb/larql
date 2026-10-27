#!/usr/bin/env python3
"""Independent offline D1 arithmetic; no model, solver or Claude analysis import.

The brief supplied expected headlines. This is not a blind freeze. Validate
hashes first; derive quantities from captured keys and per-token log-probs.
M3 stores errors only: verify their thresholds, never pretend to reconstruct
an unarchived served output. Precision tolerances are declared before replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORDERS = ("subject_after_relation", "subject_before_relation")
ABS_TOL, REL_TOL = 2e-6, 1e-5


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(test, why):
    if not test:
        raise ValueError(why)


def auc(scores, labels):
    positive = [x for x, y in zip(scores, labels) if y]
    negative = [x for x, y in zip(scores, labels) if not y]
    if min(len(positive), len(negative)) < 3:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        result[order[first:last]] = (first + 1 + last) / 2
        first = last
    return result


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else None


def classify(dose, shift, flips, c1):
    if dose < .5:
        return "K"
    if shift < 2:
        return "V"
    both = all(flips.get(order, 0) >= 1 for order in ORDERS)
    if both and c1 == "FAIL":
        return "S"
    if both and c1 == "PASS":
        return "P"
    return "U"


def verify(root):
    checked = 0
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split(None, 1)
        require(sha(root.parent / name) == expected, f"Manifest SHA mismatch: {name}")
        checked += 1
    plan = read(root / "plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    require(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == plan["plan_sha256"], "Plan digest mismatch")
    require(sha(REPO / "tools/rome_solver/rome_solver/covariance.py") == plan["services"][0]["covariance_py_sha256"], "Covariance source changed")
    records = {}
    for i, service in enumerate(plan["services"]):
        path = root / "services" / str(i)
        status, record = read(path / "status.json"), read(path / "records.json")
        require(status["ok"] and sha(path / "records.json") == status["records_sha256"], f"Record/status mismatch {i}")
        require(status["forwards"] == record["forwards"] == service["forwards"] <= 400, f"Forward count mismatch {i}")
        require(record["plan_sha256"] == plan["plan_sha256"] and record["service"] == i, f"Service identity mismatch {i}")
        if i:
            require(sha(path / "keys.npz") == record["keys_sha256"], f"Key SHA mismatch {i}")
            spec = Path(plan["inputs"]["b2v2_root"]) / (service["fact_id"] + "-L4") / "run/spec.json"
            require(sha(spec) == service["spec_sha256"], f"Spec hash mismatch {i}")
            for edit in service["edits"]:
                for name, key in (("U0.npy", "U0_sha256"), ("V0.npy", "V0_sha256"), ("k_star.npy", "k_star_sha256")):
                    require(sha(Path(edit["edit"]) / name) == edit[key], f"Edit SHA mismatch {i} {name}")
                case = f"{service['fact_id']}-L{edit['layer']}"
                verdict = Path(plan["inputs"]["b2v2_root"]) / case / "run/verdict.json"
                require(sha(verdict) == edit["b2"]["verdict_sha256"], f"B2 label verdict changed {case}")
        records[i] = record
    require(sum(r["forwards"] for r in records.values()) == plan["forwards_total"], "Total forward count mismatch")
    for layer, expected in records[0]["covariance_sha256"].items():
        require(sha(root / f"covariance/W{layer}.npy") == expected, f"W covariance SHA mismatch {layer}")
    return plan, records, checked


def vectors(root, plan):
    """Six independent float64 linear solves, batched over the five keys.

    C uses the stored actual float32 vector. Independently reconstructed C
    is checked against it; I/W are cast to the same float32 adapter format.
    """
    result, diagnostics = {}, {}
    services = plan["services"][1:]
    for layer in plan["layers"]:
        paths = [Path(next(e["edit"] for e in s["edits"] if e["layer"] == layer)) for s in services]
        keys = np.stack([np.load(p / "k_star.npy", allow_pickle=False) for p in paths], axis=1).astype(np.float64)
        matrices = {"C": paths[0] / "C.npy", "W": root / f"covariance/W{layer}.npy"}
        baseline_sha = sha(matrices["C"])
        require(all(sha(p / "C.npy") == baseline_sha for p in paths), "C differs across facts")
        direction = {"I": keys}
        for arm, path in matrices.items():
            matrix = np.load(path, allow_pickle=False).astype(np.float64)
            require(matrix.shape == (keys.shape[0], keys.shape[0]) and np.isfinite(matrix).all(), "Covariance shape/finiteness")
            require(np.allclose(matrix, matrix.T, rtol=0, atol=1e-7), "Covariance not symmetric")
            np.linalg.cholesky(matrix)  # validity check, not reused solver code
            direction[arm] = np.linalg.solve(matrix, keys)
            del matrix
        for j, (service, path) in enumerate(zip(services, paths)):
            case = f"{service['fact_id']}-L{layer}"
            reconstructed = {arm: (q[:, j] / np.dot(keys[:, j], q[:, j])).astype(np.float32) for arm, q in direction.items()}
            stored = np.load(path / "V0.npy", allow_pickle=False)[:, 0]
            require(np.allclose(stored, reconstructed["C"], atol=2e-5, rtol=2e-5), f"C vector disagrees with independent solve {case}")
            reconstructed["C"] = stored
            result[case] = {arm: v.astype(np.float64) for arm, v in reconstructed.items()}
            diagnostics[case] = {arm: float(np.dot(v, keys[:, j])) for arm, v in result[case].items()}
        print(json.dumps({"reconstructed_layer": layer}), flush=True)
    return result, diagnostics


def calculate(root, plan, records, vs):
    rows, pairs, random_shifts, m3_checks = [], [], [], []
    for i, service in enumerate(plan["services"][1:], 1):
        record = records[i]
        spec = read(Path(plan["inputs"]["b2v2_root"]) / (service["fact_id"] + "-L4") / "run/spec.json")
        questions = {q["id"]: q for q in spec["questions"]}
        native_margin = {qid: math.fsum(v["new"]) - math.fsum(v["true"]) for qid, v in record["margins"]["native"].items()}
        for condition, data in record["margins"].items():
            for qid, values in data.items():
                require(len(values["new"]) == len(record["sequences"][qid]["new_ids"]) and len(values["true"]) == len(record["sequences"][qid]["true_ids"]), "Token/logprob count mismatch")
                require(all(math.isfinite(v) and v <= 1e-6 for values_ in values.values() for v in values_), "Invalid log probability")
        with np.load(root / f"services/{i}/keys.npz", allow_pickle=False) as keys:
            for edit in service["edits"]:
                layer, fact = edit["layer"], service["fact_id"]
                case = f"{fact}-L{layer}"
                row = dict(fact_id=fact, layer=layer, c1_b2=edit["b2"]["statuses"]["C1"], c6_b2=edit["b2"]["statuses"]["C6"], arms={})
                qids = list(service["prompts"])
                neighbour_ids = [qid for qid in ("O1", "O2", "O3") if not set(questions[qid]["expect"]["accept"]) & set(spec["new"]["accept"])]
                for arm in plan["arms"]:
                    dose, maxima, argmax = {}, {}, {}
                    for qid in qids:
                        captured = keys[f"{qid}:L{layer}"]
                        require(captured.dtype == np.float32 and captured.ndim == 2 and np.isfinite(captured).all(), "Bad captured keys")
                        all_dose = captured.astype(np.float64) @ vs[case][arm]
                        maxima[qid], argmax[qid] = float(np.max(all_dose)), int(np.argmax(all_dose))
                        position = record["positions"][qid]
                        if position is not None:
                            dose[qid] = float(all_dose[position])
                    condition = f"{arm}:L{layer}"
                    margins = {qid: math.fsum(v["new"]) - math.fsum(v["true"]) for qid, v in record["margins"][condition].items()}
                    shifts = {qid: value - native_margin[qid] for qid, value in margins.items()}
                    flips = {order: sum(margins[q] > 0 for q in ("E1", "E2", "E3", "E4") if questions[q]["order"] == order) for order in ORDERS}
                    median_dose = statistics.median(dose[q] for q in ("E1", "E2", "E3", "E4"))
                    median_shift = statistics.median(shifts[q] for q in ("E1", "E2", "E3", "E4"))
                    require(median_dose != 0, "Undefined selectivity denominator")
                    row["arms"][arm] = dict(dose=dose, max_dose=maxima, max_position=argmax, margin=margins, shift=shifts,
                        dose_E=median_dose, shift_E=median_shift, flips=flips,
                        selectivity_index=statistics.median(dose[q] for q in neighbour_ids) / median_dose,
                        mechanical_class=classify(median_dose, median_shift, flips, row["c1_b2"]))
                random = record["margins"][f"random:L{layer}"]
                absolute_shifts = [abs(math.fsum(random[q]["new"]) - math.fsum(random[q]["true"]) - native_margin[q]) for q in ("E1", "E2", "E3", "E4")]
                row["random_abs_shift_E"] = statistics.median(absolute_shifts)
                random_shifts.extend(absolute_shifts)
                evidence = read(Path(plan["inputs"]["b2v2_root"]) / case / "run/evidence.json")
                import re
                for qid in neighbour_ids:
                    generation = next(g for g in evidence["generations"] if g["question_id"] == qid and g["condition"] == "edit")
                    label = any(re.search(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])", generation["text"]) for s in spec["new"]["accept"])
                    require(label == edit["b2"]["bleed"][qid], f"Bleed label mismatch {case} {qid}")
                    pairs.append(dict(case=case, qid=qid, dose=row["arms"]["C"]["dose"][qid], shift=row["arms"]["C"]["shift"][qid], bleed=label))
                m3 = record["m3"][f"L{layer}"]
                checks = dict(case=case, native_error=m3["native_rel"], arm_errors={a: m3[a]["rel"] for a in plan["arms"]})
                checks["reported_error_thresholds_pass"] = checks["native_error"] <= .02 and all(v <= .05 for v in checks["arm_errors"].values())
                m3_checks.append(checks)
                rows.append(row)
    positives = [r for r in rows if r["c1_b2"] == "PASS"]
    probe_counts = {f"{r['fact_id']}-L{r['layer']}": sum(r["arms"]["C"]["margin"][q] > 0 for q in ("E1", "E2", "E3", "E4")) for r in positives}
    require(len(positives) == 4, "Unexpected C1 positive set")
    valid = all(count >= 3 for count in probe_counts.values())
    effects, geometry = {}, {}
    for arm in ("I", "W"):
        low = [r for r in positives if r["arms"][arm]["selectivity_index"] <= .67 * r["arms"]["C"]["selectivity_index"]]
        high = [r for r in positives if r["arms"][arm]["selectivity_index"] >= 1.5 * r["arms"]["C"]["selectivity_index"]]
        improved = [r for r in low if all(v >= 1 for v in r["arms"][arm]["flips"].values())]
        effects[arm] = "improves" if len(improved) >= 3 else ("worse" if len(high) >= 3 else "neutral")
        geometry[arm] = dict(index_improvement_threshold_count=len(low), index_worse_threshold_count=len(high), raw_joint_improvement_count=len(improved), independent_of_M2=(len(low) < 3))
    score = auc([p["dose"] for p in pairs], [p["bleed"] for p in pairs])
    return dict(rows=rows, pairs=pairs, probe_valid=valid, probe_positive_counts=probe_counts,
        validated_classes_C={} if not valid else {f"{r['fact_id']}-L{r['layer']}": r["arms"]["C"]["mechanical_class"] for r in rows},
        random_noise_E=statistics.median(random_shifts), random_noise_ok=statistics.median(random_shifts) < 1,
        bleed_auc_C=score, bleed_reading="predicts" if score is not None and score >= .8 else ("does_not_predict" if score is not None and score <= .6 else "inconclusive"),
        bleed_pairs=dict(bleed=sum(p["bleed"] for p in pairs), kept=sum(not p["bleed"] for p in pairs)),
        dose_shift_spearman_C=spearman([p["dose"] for p in pairs], [p["shift"] for p in pairs]),
        spearman_interpretation="SUSPENDED_INVALID_M2" if not valid else "reportable",
        arm_effects=effects, arm_geometry=geometry, m3_checks=m3_checks,
        m3_ok=all(c["reported_error_thresholds_pass"] for c in m3_checks), m3_module_output_vectors_archived=False,
        m3_served_errors_independently_recomputed=False)


def compare(own, peer):
    differences, numeric = [], []
    def check(path, a, b):
        if isinstance(a, (float, int)) and not isinstance(a, bool):
            error = abs(a - b)
            numeric.append(dict(path=path, abs_error=error, exact=a == b))
            equal = math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL)
        else:
            equal = a == b
        if not equal:
            differences.append(dict(path=path, codex=a, claude=b))
    for key in ("probe_valid", "random_noise_E", "random_noise_ok", "bleed_auc_C", "bleed_reading", "bleed_pairs", "dose_shift_spearman_C", "arm_effects", "m3_ok"):
        check(key, own[key], peer[key])
    lookup = {(r["fact_id"], r["layer"]): r for r in peer["rows"]}
    for r in own["rows"]:
        archived = lookup[(r["fact_id"], r["layer"])]; case = f"{r['fact_id']}-L{r['layer']}"
        check(case + "/random_abs_shift_E", r["random_abs_shift_E"], archived["random_abs_shift_E"])
        for arm, a in r["arms"].items():
            b = archived["arms"][arm]
            for field in ("dose_E", "shift_E", "selectivity_index"):
                check(f"{case}/{arm}/{field}", a[field], b[field])
            for field in ("dose", "max_dose", "margin", "shift"):
                require(set(a[field]) == set(b[field]), f"Comparison membership differs {case} {arm} {field}")
                for qid, value in a[field].items():
                    check(f"{case}/{arm}/{field}/{qid}", value, b[field][qid])
            for order in ORDERS:
                check(f"{case}/{arm}/flips/{order}", a["flips"][order], b["flips"].get(order, 0))
            check(f"{case}/{arm}/mechanical_class", a["mechanical_class"], b["class"])
    return dict(differences=differences, numeric_comparisons=len(numeric), exact_numeric_agreements=sum(n["exact"] for n in numeric),
                max_abs_error=max(n["abs_error"] for n in numeric), abs_tolerance=ABS_TOL, rel_tolerance=REL_TOL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plan, records, checked = verify(args.root)
    v, normalization = vectors(args.root, plan)
    result = calculate(args.root, plan, records, v)
    result.update(author="Codex", model_forwards=0, claimed_blind_freeze=False, analysis_code_imported=False,
                  plan_sha256=plan["plan_sha256"], actual_service_forwards=sum(r["forwards"] for r in records.values()),
                  verified_manifest_files=checked, vector_normalization=normalization, script_sha256=sha(__file__))
    result["comparison"] = compare(result, read(args.root / "analysis.json"))
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("probe_valid", "probe_positive_counts", "bleed_auc_C", "bleed_pairs", "arm_effects", "m3_ok", "comparison")}))


if __name__ == "__main__":
    main()
