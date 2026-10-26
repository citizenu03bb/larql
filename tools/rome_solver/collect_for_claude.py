#!/usr/bin/env python3
"""Model-free inventory of completed primary edits, never a certification.

Input membership/order comes only from B0's five public facts and L4/L8/L12.
This produces a handoff for the independent harness author, not scores or
guesses at C1/C2. Failed/missing edits are retained as such and never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAYERS = (4, 8, 12)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    cost = json.loads((REPO / "docs/research-data/phase-b0-preflight/budget.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    b1 = json.loads((root / "b1/status.json").read_text())
    if b1["status"] != "complete" or b1["synthetic"] or b1["forwards"] != 200:
        raise ValueError("Real B1 is not complete")
    if b1["checkpoint_sha256"] != preflight["checkpoint_sha256"]:
        raise ValueError("B1 checkpoint mismatch")
    result = dict(certification_owner="Claude Code", inventory_only=True,
                  protocol="phase-b-rome-b1-b2-v1", checkpoint_sha256=preflight["checkpoint_sha256"],
                  selection=None, blind_set_opened=False, cases=[], ready=0, pending=0, failed=0)
    for fact, expected in sorted(cost["facts"].items()):
        spec = REPO / f"tools/edit_cert/facts/phase-b-dev/{fact}.json"
        if sha(spec) != expected["spec_sha256"]:
            raise ValueError("Public dev spec changed")
        for layer in LAYERS:
            folder = root / f"solves/{fact}-L{layer}"
            item = dict(fact_id=fact, layer=layer, spec=str(spec), spec_sha256=sha(spec),
                        folder=str(folder), status="pending", C1=None, C2=None)
            if (folder / "status.json").exists():
                status = json.loads((folder / "status.json").read_text())
                item["status"] = status["status"]
                if status["status"] == "complete":
                    if status["synthetic"] or status["checkpoint_sha256"] != preflight["checkpoint_sha256"]:
                        raise ValueError("Edit synthetic/checkpoint mismatch")
                    if (status["forwards"], status["gradient_steps"], status["fact_id"], status["layer"]) != (130, 25, fact, layer):
                        raise ValueError("Edit count/order mismatch")
                    edit = folder / "edit"
                    if sha(edit / "solver.json") != status["edit_manifest_sha256"]:
                        raise ValueError("Edit manifest differs from service completion receipt")
                    metadata = json.loads((edit / "solver.json").read_text())
                    for name, expected_sha in metadata["artifact_sha256"].items():
                        if sha(edit / name) != expected_sha:
                            raise ValueError(f"Edit artifact mismatch: {fact} L{layer} {name}")
                    if sha(edit / "C.npy") != b1["covariance_sha256"][str(layer)]:
                        raise ValueError("Saved edit C differs from B1")
                    item.update(edit=str(edit), edit_content_sha256=status["edit_content_sha256"],
                                solver_sha256=sha(edit / "solver.json"),
                                independent_algebra_inputs=["k_star.npy", "v_star.npy", "w_k_star.npy", "C.npy"],
                                plan_command=["python3", "-m", "edit_cert", "plan", "--run", f"<cert-root>/{fact}-L{layer}",
                                              "--fact", str(spec), "--edit", str(edit), "--instrument", "nf4-e2b", "--cap", "400"])
                    result["ready"] += 1
                elif status["status"] == "failed":
                    result["failed"] += 1
                    item["error"] = status.get("error")
                else:
                    result["pending"] += 1
            else:
                result["pending"] += 1
            result["cases"].append(item)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = inventory(a.root)
    with a.out.open("x") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("ready", "pending", "failed")}))


if __name__ == "__main__":
    main()
