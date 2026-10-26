"""A sufficient refusal witness must not pretend to prove certification."""
import importlib.util
from pathlib import Path

module = importlib.util.spec_from_file_location("refusal_witnesses", Path(__file__).parents[1] / "xcheck_b2_refusal_witnesses.py")
proof = importlib.util.module_from_spec(module)
module.loader.exec_module(proof)


def fixture():
    return dict(controls={"C1": "PASS", "C2": "PASS"}, reported_controls={"C1": "PASS", "C2": "PASS"},
                questions=[dict(control="attribute", qualified=True, order=o) for o in proof.ORDERS])


def test_positive_c1_c2_cannot_prove_full_certification_or_refusal():
    assert proof.witness(fixture()) is None


def test_agreed_mandatory_failure_is_sufficient():
    case = fixture()
    case["controls"]["C1"] = case["reported_controls"]["C1"] = "FAIL"
    assert proof.witness(case)["control"] == "C1"


def test_coverage_witness_is_separate_from_disputed_c2_status():
    case = fixture()
    case["controls"]["C2"] = "NOT_EVALUATED"
    case["questions"].pop()
    assert proof.witness(case)["control"] == "C3"
    assert proof.witness(case)["independently_recomputed"] == "NOT_PASS"
