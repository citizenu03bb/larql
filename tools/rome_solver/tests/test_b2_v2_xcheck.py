"""Logical control tests with planted data; no model/harness imports."""
import importlib.util
from pathlib import Path

module = importlib.util.spec_from_file_location("b2v2", Path(__file__).parents[1] / "xcheck_b2_v2.py")
checker = importlib.util.module_from_spec(module)
module.loader.exec_module(checker)


def fixture():
    questions = [dict(id=f"{prefix}{i}", control=control, order=order, **extra)
                 for prefix, control, extra in (("E", "efficacy", {}), ("A", "attribute", {"expect": {"accept": ["nine"], "reject": ["ten"]}}))
                 for i, order in enumerate(sorted(checker.ORDERS))]
    questions += [dict(id=q, control=c, order="none", expect=dict(accept=[a], reject=r)) for q, c, a, r in
                  (("W", "answer_only", "Beethoven", ["Italian"]), ("O", "off_target", "Polish", ["Italian"]), ("P", "propagation", "Italian", []))]
    spec = dict(old={"accept": ["German"]}, new={"accept": ["Italian"]}, min_efficacy=2, swap_markers=[], questions=questions)
    generations, boundaries = [], []
    for q in questions:
        for condition in ("native", "edit"):
            text = ("German" if condition == "native" else "Italian") if q["control"] == "efficacy" else q["expect"]["accept"][0]
            generations.append(dict(question_id=q["id"], condition=condition, text=text, ids=[7, 106], eos=True, capped=False))
        if q["control"] == "efficacy":
            for condition in ("random", "reloaded"):
                generations.append(dict(question_id=q["id"], condition=condition, text="German" if condition == "random" else "Italian", ids=[7, 106], eos=True, capped=False))
            for condition, margin in (("native", -3.), ("edit", 3.), ("random", -2.)):
                boundaries.append(dict(question_id=q["id"], condition=condition, native_id=7, target_id=9, top1=7 if margin < 0 else 9, margin=margin, partial=False, variant=""))
    evidence = dict(generations=generations, boundaries=boundaries, kv_write_deadline=13, artifact_sha256={"edit": "same", "reloaded": "same"})
    backend = dict(kind="low_rank", params=dict(targets=[dict(layer=8)]), form=dict(answer_time_trigger=False, donor_pass=False, uses_native_proposal=False, substitutes_subject_tokens=False, single_artifact=True, applied_every_forward=True))
    plan = dict(services=[dict(index=0, phase="edit"), dict(index=1, phase="reload")])
    return spec, evidence, backend, plan


def generation(args, qid, condition="edit"):
    return next(g for g in args[1]["generations"] if (g["question_id"], g["condition"]) == (qid, condition))


def test_positive_can_certify_but_donor_and_propagation_are_not_mandatory():
    result = checker.replay(*fixture())
    assert result["verdict"] == "CERTIFIED"
    assert result["controls"]["C5"] == "NOT_EVALUATED"
    assert result["controls"]["C8"] == "PASS"


def test_capped_preservation_is_unknown_not_a_pass_or_observed_value_failure():
    args = fixture()
    generation(args, "A0").update(eos=False, capped=True)
    result = checker.replay(*args)
    assert result["controls"]["C1"] == "PASS"
    assert result["controls"]["C2"] == result["controls"]["C3"] == "NOT_EVALUATED"
    assert result["verdict"] == "REFUSED"


def test_answer_only_entity_substitution_fails_and_rephrased_off_target_value_passes():
    args = fixture()
    generation(args, "W")["text"] = "Giuseppe Verdi composed it."
    generation(args, "O")["text"] = "He was Polish."
    result = checker.replay(*args)
    assert result["controls"]["C3"] == "FAIL"
    assert result["controls"]["C6"] == "PASS"


def test_random_flip_and_reload_mismatch_cannot_certify():
    args = fixture()
    generation(args, "E0", "random")["text"] = "Italian"
    generation(args, "E0", "reloaded")["ids"] = [8, 106]
    result = checker.replay(*args)
    assert result["controls"]["C4"] == result["controls"]["C7"] == "FAIL"


def test_insufficient_margin_ratio_fails_even_when_random_does_not_flip():
    args = fixture()
    for b in args[1]["boundaries"]:
        if b["condition"] == "random":
            b["margin"] = 1.
    assert checker.replay(*args)["controls"]["C4"] == "FAIL"


def test_native_cap_removes_order_coverage_before_edit_scoring():
    args = fixture()
    generation(args, "A0", "native").update(eos=False, capped=True)
    result = checker.replay(*args)
    assert not result["feasibility"]["feasible"]
    assert result["controls"]["C2"] == "NOT_EVALUATED"
