#!/usr/bin/env python3
"""Serial execution of the registered 15 primary slots; stop at any failure.

No model/method imports, certification, layer search or automatic retry.
Already completed slots are hash-verified and reused, never rerun. Every new
slot goes through the committed rehearsal/flock/bounded-service launcher.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from collect_for_claude import inventory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def write(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def host_snapshot():
    ps = subprocess.check_output(["ps", "-eo", "pid=,comm=,rss="], text=True)
    heavy = []
    for row in ps.splitlines():
        pid, comm, rss = row.split(maxsplit=2)
        if comm.startswith("python") and int(rss) > 4 * 1024**2:
            heavy.append(dict(pid=int(pid), comm=comm, rss_kib=int(rss)))
    locks = json.loads(subprocess.check_output(["lslocks", "-J", "-o", "PID,COMMAND,PATH"], text=True))
    shared = [x for x in locks["locks"] if (x.get("path") or "").endswith("/.heavy-model.lock")]
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout.strip()
    if heavy and not shared:
        raise RuntimeError("Potential heavy Python worker outside the shared lock; stop for inspection")
    return dict(heavy_python_processes=heavy, shared_lock_holders=shared, gpu_processes=gpu)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-approved", action="store_true")
    a = p.parse_args()
    report = inventory(a.root)
    write(a.root / "certification-handoff-latest.json", report)
    if not a.run_approved:
        print(json.dumps({k: report[k] for k in ("ready", "pending", "failed")}))
        return 0
    if report["failed"]:
        raise RuntimeError("A failed primary slot exists: no automatic retry")
    state = dict(status="running", total=15, completed=report["ready"], certification_owner="Claude Code")
    try:
        for case in report["cases"]:
            if case["status"] == "complete":
                continue
            if case["status"] != "pending":
                raise RuntimeError("A primary slot is already active; no overlapping launch")
            state.update(current_fact=case["fact_id"], current_layer=case["layer"], current_folder=case["folder"])
            write(a.root / "b2-batch-state.json", state)
            folder = Path(case["folder"])
            folder.parent.mkdir(parents=True, exist_ok=True)
            write(folder.parent / (folder.name + "-host-before.json"), host_snapshot())
            command = [sys.executable, str(HERE / "launch_phase_b.py"), "--stage", "b2",
                       "--fact-id", case["fact_id"], "--layer", str(case["layer"]),
                       "--covariance", str(a.root / "b1"), "--out", str(folder),
                       "--preflight", str(a.root / "preflight.json"), "--approved-current-protocol"]
            with (folder.parent / (folder.name + "-launcher.log")).open("x") as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Service failed for {case['fact_id']} L{case['layer']}: {result.returncode}")
            updated = inventory(a.root)
            state["completed"] = updated["ready"]
            write(a.root / "certification-handoff-latest.json", updated)
            immutable = a.root / f"certification-handoff-{updated['ready']:02}.json"
            with immutable.open("x") as f:
                f.write(json.dumps(updated, indent=2, sort_keys=True) + "\n")
            print(json.dumps(dict(completed=updated["ready"], total=15, fact=case["fact_id"], layer=case["layer"])), flush=True)
        state.update(status="solves_complete_pending_independent_certification", completed=15)
        write(a.root / "b2-batch-state.json", state)
        return 0
    except BaseException as exc:
        state.update(status="stopped", error=str(exc), time_ns=time.time_ns())
        write(a.root / "b2-batch-state.json", state)
        raise


if __name__ == "__main__":
    sys.exit(main())
