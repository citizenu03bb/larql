"""The explicit amendment must not erase a failure or change the denominator."""
import copy
import importlib.util
import json
from pathlib import Path

module = importlib.util.spec_from_file_location("c2_arbitration", Path(__file__).parents[1] / "apply_b2_c2_arbitration.py")
gate = importlib.util.module_from_spec(module)
module.loader.exec_module(gate)


def fixture():
    facts = json.loads((gate.REPO / "docs/research-data/phase-b0-preflight/budget.json").read_text())["facts"]
    cases = {f"{fact}-L{layer}": dict(reported_controls={"C1": "FAIL", "C2": "PASS"},
              questions=[dict(control="attribute", order=order, qualified=True) for order in gate.ORDERS])
             for fact in facts for layer in (4, 8, 12)}
    return dict(cases=cases, differences=[], pending=[], peer_complete=True)


def test_missing_order_only_downgrades_a_reported_pass_without_mutating_input():
    source = fixture()
    case = next(iter(source["cases"]))
    source["cases"][case]["reported_controls"]["C1"] = "PASS"
    source["cases"][case]["questions"].pop()
    original = copy.deepcopy(source)
    result = gate.decide(source)
    assert result["cases"][case]["decision_controls"]["C2"] == "NOT_EVALUATED"
    assert result["gate"] == "STOP_BELOW_2_OF_5"
    assert source == original
    source["cases"][case]["reported_controls"]["C2"] = "FAIL"
    assert gate.decide(source)["cases"][case]["decision_controls"]["C2"] == "FAIL"


def test_registered_denominator_tie_break_and_threshold():
    source = fixture()
    facts = sorted({c.rsplit("-L", 1)[0] for c in source["cases"]})
    for fact in facts[:2]:
        for layer in (4, 8, 12):
            source["cases"][f"{fact}-L{layer}"]["reported_controls"]["C1"] = "PASS"
    result = gate.decide(source)
    assert result["pass_counts"] == {"4": 2, "8": 2, "12": 2}
    assert result["selected_layer"] == 4


def test_unarbitrated_difference_blocks_positive_continuation():
    source = fixture()
    for value in source["cases"].values():
        value["reported_controls"]["C1"] = "PASS"
    source["differences"] = [dict(case=next(iter(source["cases"])), control="C1", codex="FAIL", claude="PASS")]
    assert gate.decide(source)["gate"] == "PENDING_OTHER_ARBITRATIONS"
