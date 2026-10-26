#!/usr/bin/env python3
"""Independent, model-free R1--R5 replay from Pat's certification brief.

Uses only NumPy and the standard library. Does not import archived runners,
their checkers, or the certification harness. All paths recorded in the output
are relative to --archive-root. Missing inputs or supplied digest mismatches
abort before an output is written. No inference, subprocess, or network calls.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ASYM = "nf4-asymmetry-20261026/interventions"
STATIC = "nf4-phase3-static-20261026/calib"
SPLIT = "nf4-phase3b-rows-20261026"
Q4K = "entity-first-attribute-swap-20261025"
NE = "NOT_EVALUATED"
REQUIRED = ("C0", "C1", "C2", "C3", "C4", "C6", "C7")


def margin(logits, native, target):
    return float(np.subtract(np.float32(logits[target]),
                             np.float32(logits[native]), dtype=np.float32))


def increment(edited, native):
    return float(np.subtract(np.float32(edited), np.float32(native),
                             dtype=np.float32))


def rank(logits, token):
    return 1 + int(np.count_nonzero(logits > logits[token]))


def cast_norm(vector):
    # Measure the ACTUAL float16 vector in float64, not an overflowing f16 sum.
    return float(np.linalg.norm(vector.astype(np.float16).astype(np.float64)))


class Archive:
    def __init__(self, root):
        self.root = root.resolve()
        self.sources = {}
        self.json_cache = {}
        self.arrays = {}

    def file(self, name, expected=None):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Archive path escapes root: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"SHA-256 mismatch: {name}: {digest} != {expected}")
        record = self.sources.setdefault(name, {"sha256": digest})
        if expected is not None:
            record.update(archived_sha256=expected, sha256_matches=True)
        if path.suffix == ".npz":
            record.setdefault("archived_sha256", None)
            record.setdefault("sha256_matches", None)
        return path

    def document(self, name):
        return self.file(name).read_text()

    def read_json(self, name):
        if name not in self.json_cache:
            self.json_cache[name] = json.loads(self.document(name))
        return self.json_cache[name]

    def array(self, name):
        result = np.load(self.file(name), allow_pickle=False)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Nonfinite array: {name}")
        return result

    def npz(self, name, expected=None):
        # Always check a supplied reference, including on cached reads.
        path = self.file(name, expected)
        if name not in self.arrays:
            with np.load(path, allow_pickle=False) as z:
                self.arrays[name] = {key: z[key] for key in z.files}
            logits = self.arrays[name]["logits"]
            if logits.dtype != np.float32 or logits.ndim != 1:
                raise ValueError(f"Expected flat float32 logits: {name}")
            if not np.all(np.isfinite(logits)):
                raise ValueError(f"Nonfinite logits: {name}")
        return self.arrays[name]

    def nf4(self, folder, name):
        summary = self.read_json(f"{folder}/summary.json")
        matches = [c for c in summary["calls"] if c["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one archived call: {folder}/{name}")
        call = matches[0]
        data = self.npz(f"{folder}/{call['artifact']}", call.get("artifact_sha256"))
        return data, call

    def q4k(self, ordinal):
        candidates = sorted((self.root / Q4K / "results").glob(f"{ordinal:03d}-*.json"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one Q4K job at ordinal {ordinal}")
        job = self.read_json(str(candidates[0].relative_to(self.root)))
        step = job["steps"][-1]
        name = f"{Q4K}/results/{step['artifact']}"
        # These jobs provide no NPZ digest. Do not mistake binary/contrast
        # checksums in session.json for checksums of the result arrays.
        return self.npz(name, step.get("artifact_sha256")), job


def case(form, interpretations):
    return {"verdict": None, "controls": {}, "numbers": {},
            "archived_comparisons": [], "observations": [],
            "form": form, "interpretations": interpretations}


def number(c, key, actual, reference, source, transform=None):
    if key in c["numbers"]:
        raise ValueError(f"Duplicate number: {key}")
    c["numbers"][key] = actual
    normalized = reference + 1 if transform == "zero_based_rank_plus_one" else reference
    c["archived_comparisons"].append({
        "number": key, "computed": actual, "archived": reference,
        "normalized_reference": normalized, "exact_equal": actual == normalized,
        "raw_exact_equal": actual == reference, "source": source,
        "reference_transform": transform,
    })


def observe(c, question, role, base, edit, native, target=None,
            variant="complete", mention="before", source=None):
    baseline = int(np.argmax(base))
    c["observations"].append({
        "question": question, "role": role, "seen_formulation": role == "efficacy",
        "variant": variant, "mention_group": mention, "source": source,
        "native_id": native, "target_id": target, "native_top1": baseline,
        "edit_top1": int(np.argmax(edit)), "qualified": baseline == native,
        "evidence": "resumed_teacher_forced_frontier",
    })


def judge(c):
    form = c["form"]
    defects = [key for key in ("response_time_trigger", "donor_pass",
                              "native_proposal", "subject_substitution") if form[key]]
    defects += [key for key in ("unique_stored_artifact", "applied_each_forward")
                if not form[key]]
    if form["layer_written"] is not None and form["layer_written"] > 22:
        defects.append("write_after_block_22")
    controls = {f"C{i}": NE for i in range(9)}
    controls["C0"] = "FAIL" if defects else "PASS"
    witnesses = {key: [] for key in controls}
    witnesses["C0"] = defects
    mention_groups = {"before": NE, "after": NE}
    for o in c["observations"]:
        if not o["qualified"]:
            continue
        if o["role"] == "efficacy" and o["variant"] == "complete":
            if o["edit_top1"] != o["target_id"]:
                controls["C1"] = "FAIL"
                witnesses["C1"].append(o)
                if not o["seen_formulation"]:
                    controls["C3"] = "FAIL"
                    witnesses["C3"].append(o)
                    mention_groups[o["mention_group"]] = "FAIL"
        elif o["role"] == "attribute" and o["edit_top1"] != o["native_id"]:
            controls["C2"] = controls["C3"] = "FAIL"
            witnesses["C2"].append(o)
            witnesses["C3"].append(o)
            mention_groups[o["mention_group"]] = "FAIL"
        elif o["role"] == "off_target" and o["edit_top1"] != o["native_top1"]:
            controls["C6"] = "FAIL"
            witnesses["C6"].append(o)
    # All efficacy formulations in this fixed composition were seen. Therefore
    # C4 has an empty domain. C5/C8 are nonblocking observations, with no
    # negative criterion supplied here. No reloaded generation exists for C7.
    assert all(o["seen_formulation"] for o in c["observations"] if o["role"] == "efficacy")
    assert all(v in ("FAIL", NE) for k, v in controls.items() if k != "C0")
    c.update(controls=controls, control_failures=witnesses,
             mention_groups=mention_groups,
             verdict="CERTIFIED" if all(controls[k] == "PASS" for k in REQUIRED) else "REFUSED")


def build(root):
    a = Archive(root)
    asym = a.read_json(f"{ASYM}/summary.json")
    static = a.read_json(f"{STATIC}/summary.json")
    split = a.read_json(f"{SPLIT}/summary.json")
    qsummary = a.read_json(f"{Q4K}/summary.json")
    session = a.read_json(f"{Q4K}/session.json")
    # Independent sources for scalar comparisons and frozen protocol provenance.
    av = a.read_json(f"{ASYM}/verification.json")
    sv = a.read_json("nf4-phase3-static-20261026/verification.json")
    pv = a.read_json(f"{SPLIT}/verification.json")
    qv = a.read_json(f"{Q4K}/verification.json")
    a.document(f"{Q4K}/report.md")
    for doc, fragments in (
        (asym["protocol"], ("Derive this\nonce on the symbol question", "saved native value boundaries")),
        (static["protocol"], ("no donor pass, no answer-time trigger", "at every forward", "post-block L21")),
        (split["protocol"], ("row 5 only", "row 30 only", "post-block L21")),
        (session["preregistration"], ("native proposal", "once at L21 row 5, then release")),
    ):
        if not all(f in doc for f in fragments):
            raise ValueError("Frozen protocol evidence no longer matches the interpreted form")

    pulse = dict(response_time_trigger=True, donor_pass=False, native_proposal=True,
                 subject_substitution=False, unique_stored_artifact=True,
                 applied_each_forward=False, layer_written=21,
                 protocol_source=f"{ASYM}/summary.json#/protocol")
    c1 = case(dict(pulse, artifact=f"{ASYM}/residual.npy"), [
        "A stored single-pulse vector, scheduled at a native proposal, fails persistent form.",
        "Donor construction/calibration is distinguished from an additional donor inference required to apply the fixed vector.",
        "A unique artifact means the fixed numerical vector with frozen parameters, not a deployable patch bundle.",
    ])
    c2 = case(dict(response_time_trigger=False, donor_pass=False, native_proposal=False,
                   subject_substitution=False, unique_stored_artifact=True,
                   applied_each_forward=True, layer_written=21,
                   artifact=f"{ASYM}/residual.npy", gain=1.0,
                   protocol_source=f"{STATIC}/summary.json#/protocol"), [
        "The edit is the fixed static rule at g*=1.0, not the measurement procedure used for these resumed frontiers.",
        "A lexical rule applied every forward is not an answer-time schedule, including when it matches generated Hydrogen.",
        "Only row5 and row30 at gain 1 are partial witnesses; both_half and other calibration gains are outside this case.",
        "Complete frontiers are grouped by question subject/relation order (before); row30 partials belong to after.",
        "Each 3b increment uses its own 3b native boundary, never the calibration baseline.",
    ])
    c3 = case(dict(pulse, artifact=f"{ASYM}/random.npy"), [
        "The random vector is the edit itself. C4 is still restricted to unseen efficacy formulations.",
        "Norms are measured in float64 after casting each stored vector to float16; tolerance is relative to residual norm.",
        "The random NPZ requested vector is the float16 round-trip of the stored float32 vector; raw float32 equality is not expected.",
    ])
    c4 = case(dict(response_time_trigger=True, donor_pass=True, native_proposal=True,
                   subject_substitution=True, unique_stored_artifact=False,
                   applied_each_forward=False, layer_written=None,
                   protocol_source=f"{ASYM}/summary.json#/protocol"), [
        "The observed donor replaces the prompt subject at the saved native answer boundary; it is not a reusable residual patch.",
        "No residual layer is written by subject substitution; the donor forward is the edit itself, never a C5 ceiling.",
    ])
    c5 = case(dict(pulse, artifact=f"{Q4K}/delta.npy", gain=session["gain"],
                   protocol_source=f"{Q4K}/session.json#/preregistration"), [
        "Uses the archived precise float32 gain (the brief's 2.585 is rounded), once at row5/L21.",
        "Only the two specified attribute questions belong to R5; the optional archived symbol random is not efficacy evidence here.",
        "Q4K archived helium_rank is zero-based. The mandated strict-greater rank is one-based; both raw and normalized comparisons are retained.",
    ])
    cases = dict(R1=c1, R2=c2, R3=c3, R4=c4, R5=c5)

    for question, native, target, role in (
        ("symbol", 236814, 2209, "efficacy"), ("number", 886, 1156, "attribute")
    ):
        data = {kind: a.nf4(ASYM, f"{question}_{kind}")[0]
                for kind in ("base", "residual", "random", "donor")}
        logits = {k: v["logits"] for k, v in data.items()}
        m = {k: margin(v, native, target) for k, v in logits.items()}
        metrics = asym["metrics"][question]
        for field, value in (("baseline_margin", m["base"]),
                             ("residual_margin", m["residual"]),
                             ("residual_increment", increment(m["residual"], m["base"])),
                             ("random_increment", increment(m["random"], m["base"])),
                             ("residual_argmax", int(np.argmax(logits["residual"])))):
            key = f"{question}_{field}"
            number(c1, key, value, metrics[field], f"{ASYM}/summary.json#/metrics/{question}/{field}")
            assert av["metrics"][question][field] == metrics[field]
        number(c3, f"{question}_random_increment", increment(m["random"], m["base"]),
               metrics["random_increment"], f"{ASYM}/summary.json#/metrics/{question}/random_increment")
        number(c4, f"{question}_donor_argmax", int(np.argmax(logits["donor"])),
               metrics["donor_argmax"], f"{ASYM}/summary.json#/metrics/{question}/donor_argmax")
        for c, kind in ((c1, "residual"), (c3, "random"), (c4, "donor")):
            observe(c, question, role, logits["base"], logits[kind], native, target,
                    source=f"{ASYM}/{question}_{kind}")
        c1.setdefault("donor_ceiling_observations", {})[question] = {
            "role": role, "top1": int(np.argmax(logits["donor"])), "margin": m["donor"],
            "counts_for_C5": role == "efficacy", "status": NE,
        }

        b, bc = a.nf4(STATIC, f"{question}_native")
        e, ec = a.nf4(STATIC, f"{question}_g1.0")
        record = next(r for r in static["calibration"]["rows"]
                      if r["question"] == question and r["gain"] == 1.0)
        ref_index = static["calibration"]["rows"].index(record)
        for field, value in (("margin", margin(e["logits"], native, target)),
                             ("argmax", int(np.argmax(e["logits"])))):
            number(c2, f"{question}_g1.0_{field}", value, record[field],
                   f"{STATIC}/summary.json#/calibration/rows/{ref_index}/{field}")
            assert sv["calibration"]["rows"][ref_index][field] == record[field]
        observe(c2, question, role, b["logits"], e["logits"], native, target,
                source=f"{STATIC}/{ec['artifact']}")
        number(c2, f"{question}_calibration_native_margin", margin(b["logits"], native, target),
               record["native_margin"], f"{STATIC}/summary.json#/calibration/rows/{ref_index}/native_margin")
        pb, pbc = a.nf4(SPLIT, f"{question}_native")
        pm = margin(pb["logits"], native, target)
        number(c2, f"{question}_3b_native_margin", pm, split["results"][question]["native"]["margin"],
               f"{SPLIT}/summary.json#/results/{question}/native/margin")
        for variant, mention in (("row5", "before"), ("row30", "after")):
            pe, pec = a.nf4(SPLIT, f"{question}_{variant}")
            ref = split["results"][question][variant]
            for field, value in (("argmax", int(np.argmax(pe["logits"]))),
                                 ("increment", increment(margin(pe["logits"], native, target), pm))):
                number(c2, f"{question}_{variant}_{field}", value, ref[field],
                       f"{SPLIT}/summary.json#/results/{question}/{variant}/{field}")
                assert pv["results"][question][variant][field] == ref[field]
            observe(c2, question, role, pb["logits"], pe["logits"], native, target,
                    variant=variant, mention=mention, source=f"{SPLIT}/{pec['artifact']}")

    for i, question in enumerate(("lithium", "carbon")):
        base, _ = a.nf4(ASYM, f"{question}_base")
        native = int(np.argmax(base["logits"]))
        for c, kind in ((c1, "residual"), (c3, "random")):
            edit, call = a.nf4(ASYM, f"{question}_{kind}")
            field = f"{kind}_argmax"
            number(c, f"{question}_{field}", int(np.argmax(edit["logits"])), asym["off_target"][i][field],
                   f"{ASYM}/summary.json#/off_target/{i}/{field}")
            observe(c, question, "off_target", base["logits"], edit["logits"], native,
                    source=f"{ASYM}/{call['artifact']}")

    residual = a.array(f"{ASYM}/residual.npy")
    random = a.array(f"{ASYM}/random.npy")
    rn, gn = cast_norm(residual), cast_norm(random)
    number(c3, "residual_float16_norm", rn, asym["norm_rule"]["actual_norm"],
           f"{ASYM}/summary.json#/norm_rule/actual_norm")
    requested = a.nf4(ASYM, "symbol_random")[0]["requested_residual"]
    number(c3, "random_float16_norm", gn, cast_norm(requested),
           f"{ASYM}/004-symbol_random.npz#cast_norm(requested_residual)")
    c3["norm_check"] = {"relative_difference": abs(gn - rn) / rn,
                        "tolerance": 0.001, "within_tolerance": abs(gn - rn) / rn <= 0.001,
                        "stored_random_matches_requested": bool(np.array_equal(random, requested)),
                        "stored_random_float16_roundtrip_matches_requested": bool(
                            np.array_equal(random.astype(np.float16).astype(np.float32), requested))}

    # Q4K stores the full contrast; row 5 and the archived precise gain define
    # the effective single vector. Confirm it is present, finite and shaped.
    delta = a.array(f"{Q4K}/delta.npy")
    if delta.ndim != 2 or delta.shape[1] != 2560 or delta.shape[0] <= 5:
        raise ValueError("Unexpected Q4K contrast shape")
    for idx, (question, ni, ei, target) in enumerate((
        ("number", 7, 8, 236778), ("mass", 18, 19, 236812)
    )):
        base, bj = a.q4k(ni)
        edit, ej = a.q4k(ei)
        ref = qsummary["questions"][idx]
        bm, em = margin(base["logits"], 236770, target), margin(edit["logits"], 236770, target)
        for key, value, archived, suffix in (
            ("baseline_margin", bm, ref["native_margin"], "native_margin"),
            ("edit_margin", em, ref["high"]["margin"], "high/margin"),
            ("edit_increment", increment(em, bm), ref["high"]["delta_margin"], "high/delta_margin"),
            ("edit_argmax", int(np.argmax(edit["logits"])), ref["high"]["top1"], "high/top1"),
        ):
            number(c5, f"{question}_{key}", value, archived,
                   f"{Q4K}/summary.json#/questions/{idx}/{suffix}")
        assert qv["questions"][idx]["native_margin"] == ref["native_margin"]
        assert qv["questions"][idx]["high_margin"] == ref["high"]["margin"]
        observe(c5, question, "attribute", base["logits"], edit["logits"], 236770, target,
                source=f"{Q4K}/results/{ej['steps'][-1]['artifact']}")
        if question == "mass":
            for label, data, archived, suffix in (
                ("native", base, ref["native_helium_rank"], "native_helium_rank"),
                ("edit", edit, ref["high"]["helium_rank"], "high/helium_rank"),
            ):
                number(c5, f"mass_{label}_four_rank", rank(data["logits"], target), archived,
                       f"{Q4K}/summary.json#/questions/{idx}/{suffix}", "zero_based_rank_plus_one")
        else:
            for ri, ordinal in enumerate((9, 22)):
                random_data, _ = a.q4k(ordinal)
                ref_random = qsummary["random_control"][ri]
                inc = increment(margin(random_data["logits"], 236770, target), bm)
                number(c5, f"number_random_{ref_random['seed']}_increment", inc, ref_random["delta_margin"],
                       f"{Q4K}/summary.json#/random_control/{ri}/delta_margin")

    for c in cases.values():
        judge(c)
        c["interpretations"].append(
            "Only resumed frontiers: outside C0, lack of a failure is NOT_EVALUATED, never PASS. C7 has no reload generation; C8 has no propagation criterion/evidence in this composition.")
    comparisons = [r for c in cases.values() for r in c["archived_comparisons"]]
    return {
        "author": "Codex", "frozen_before_reading_harness": True,
        "specification": "Pat's independent R1-R5 replay brief, 2026-10-26",
        "model_passes": 0, "cases": cases,
        "arithmetic": {"margin": "float32 subtraction of float32 target and native logits",
                       "increment": "float32 subtraction of edit and same-archive native margins",
                       "rank": "1 + count(logits strictly greater than the token logit)"},
        "comparison_summary": {"count": len(comparisons),
                               "normalized_exact_equal": sum(r["exact_equal"] for r in comparisons),
                               "raw_exact_equal": sum(r["raw_exact_equal"] for r in comparisons)},
        "sources": dict(sorted(a.sources.items())),
    }


def self_test():
    # Precision and ties are material to this specification.
    x = np.array([16777217.0, 16777218.0], dtype=np.float64)
    assert margin(x, 0, 1) == 2.0  # differs from subtract-in-f64-then-cast
    assert rank(np.array([3, 3, 4, 1], dtype=np.float32), 0) == 2
    assert increment(1.5, -0.5) == 2.0
    form = dict(response_time_trigger=False, donor_pass=False, native_proposal=False,
                subject_substitution=False, unique_stored_artifact=True,
                applied_each_forward=True, layer_written=21)
    c = case(form, [])
    observe(c, "symbol", "efficacy", np.array([3, 0]), np.array([0, 3]), 0, 1)
    observe(c, "symbol", "efficacy", np.array([3, 0]), np.array([3, 0]), 0, 1, variant="row5")
    observe(c, "number", "attribute", np.array([0, 3]), np.array([0, 3]), 0, 1)
    judge(c)
    assert c["controls"]["C0"] == "PASS" and c["controls"]["C1"] == NE
    assert c["controls"]["C2"] == NE  # ineligible native cannot fail a question
    observe(c, "number", "attribute", np.array([3, 0]), np.array([0, 3]), 0, 1,
            variant="row30", mention="after")
    judge(c)
    assert c["controls"]["C2"] == c["controls"]["C3"] == "FAIL"
    assert c["mention_groups"] == {"before": NE, "after": "FAIL"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path,
                        default=Path.home() / "Projects/larql-research/exploratory-natural-fact-edit")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: precision, strict rank, eligibility, complete/partial and mention grouping OK")
    if args.out is not None:
        result = build(args.archive_root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"verdicts": {k: c["verdict"] for k, c in result["cases"].items()},
                          "comparison_summary": result["comparison_summary"]}))
    elif not args.self_test:
        parser.error("--out or --self-test is required")


if __name__ == "__main__":
    main()
