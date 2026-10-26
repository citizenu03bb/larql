"""Independent gate-reading edge cases, with no model or harness imports."""
import importlib.util
from pathlib import Path

module = importlib.util.spec_from_file_location("gate_xcheck", Path(__file__).parents[1] / "xcheck_b2_gate.py")
gate = importlib.util.module_from_spec(module)
module.loader.exec_module(gate)


def fixture():
    questions = [dict(id=f"{prefix}{i}", control=control, order=order, **extra)
                 for prefix, control, extra in (("E", "efficacy", {}), ("A", "attribute", {"expect": {"accept": ["nine"], "reject": ["ten"]}}))
                 for i, order in enumerate(sorted(gate.ORDERS))]
    spec = dict(old={"accept": ["German"]}, new={"accept": ["Italian"]},
                swap_markers=["DonorName"], questions=questions, min_efficacy=2)
    generations = [dict(question_id=q["id"], condition=condition, text=("German" if condition == "native" else "Italian")
                        if q["control"] == "efficacy" else "nine", ids=[], eos=True, capped=False)
                   for q in questions for condition in ("native", "edit")]
    return spec, generations


def edit(generations, question):
    return next(g for g in generations if g["condition"] == "edit" and g["question_id"] == question)


def test_positive_both_orders():
    assert gate.controls(*fixture())[0] == {"C1": "PASS", "C2": "PASS"}


def test_capped_native_excluded_even_when_word_correct():
    spec, generations = fixture()
    generations[0].update(eos=False, capped=True)
    result, questions = gate.controls(spec, generations)
    assert result["C1"] == "NOT_EVALUATED"
    assert not questions[0]["qualified"]


def test_seen_cannot_supply_positive_coverage_but_can_fail():
    spec, generations = fixture()
    spec["questions"][0]["seen"] = True
    assert gate.controls(spec, generations)[0]["C1"] == "NOT_EVALUATED"
    edit(generations, "E0")["text"] = "German"
    assert gate.controls(spec, generations)[0]["C1"] == "FAIL"


def test_unreadable_edit_never_passes_and_explicit_reject_fails():
    spec, generations = fixture()
    edit(generations, "A0").update(eos=False, capped=True)
    assert gate.controls(spec, generations)[0]["C2"] == "NOT_EVALUATED"
    edit(generations, "A0")["text"] = "nine, ten"
    assert gate.controls(spec, generations)[0]["C2"] == "FAIL"


def test_surface_boundaries_and_case_are_preserved():
    g = {"text": "NotItalian italian Italianate"}
    assert not gate.contains(g, ["Italian"])
    g["text"] += " (Italian)."
    assert gate.contains(g, ["Italian"])


def test_no_failure_hidden_by_insufficient_coverage():
    spec, generations = fixture()
    generations[0].update(eos=False, capped=True)
    edit(generations, "E1")["text"] = "German"
    assert gate.controls(spec, generations)[0]["C1"] == "FAIL"
