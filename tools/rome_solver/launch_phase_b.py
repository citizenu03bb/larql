#!/usr/bin/env python3
"""Fresh synthetic rehearsal, then ONE approved bounded real B1/B2 service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = "/mnt/models/finetune/venv/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("b1", "b2"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--fact-id")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--covariance", type=Path)
    parser.add_argument("--approved-current-protocol", action="store_true")
    args = parser.parse_args()
    if not args.approved_current_protocol:
        parser.error("Explicit approval of the committed protocol is required")
    if args.out.exists():
        raise FileExistsError(args.out)
    committed = ["tools/rome_solver/run_phase_b.py", "tools/rome_solver/launch_phase_b.py",
                 "docs/phase-b-b1-b2-protocol.md"]
    subprocess.run(["git", "ls-files", "--error-unmatch", *committed], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO, check=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONHASHSEED="20261026")
    # The same per-stage wrapper is rehearsed; a real covariance can never
    # enter a synthetic solve. Use the synthetic B1 saved by this launch.
    dry = args.out.with_name(args.out.name + "-dry-run")
    if args.stage == "b1":
        subprocess.run([PYTHON, str(HERE / "run_phase_b.py"), "--stage", "b1", "--synthetic", "--out", str(dry)], env=env, check=True)
    else:
        dry_cov = dry.with_name(dry.name + "-covariance")
        subprocess.run([PYTHON, str(HERE / "run_phase_b.py"), "--stage", "b1", "--synthetic", "--out", str(dry_cov)], env=env, check=True)
        subprocess.run([PYTHON, str(HERE / "run_phase_b.py"), "--stage", "b2", "--synthetic", "--out", str(dry),
                        "--fact-id", str(args.fact_id), "--layer", str(args.layer), "--covariance", str(dry_cov)], env=env, check=True)
    receipt = json.loads((dry / "status.json").read_text())
    if receipt["status"] != "complete" or not receipt["synthetic"]:
        raise RuntimeError("Synthetic rehearsal did not complete")
    unit = f"phase-b-{args.stage}-{time.time_ns()}"
    worker = [PYTHON, str(HERE / "run_phase_b.py"), "--stage", args.stage, "--out", str(args.out),
              "--preflight", str(args.preflight)]
    if args.stage == "b2":
        worker += ["--fact-id", str(args.fact_id), "--layer", str(args.layer), "--covariance", str(args.covariance)]
    lock = str(Path.home() / "Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock")
    command = ["flock", "-w", "3600", lock, "systemd-run", "--user", "--wait", "--pipe", "--collect",
               "--unit=" + unit, "--property=MemoryHigh=22G", "--property=MemoryMax=24G",
               "--property=MemorySwapMax=8G", "--property=RuntimeMaxSec=1800", "--property=TimeoutStopSec=10",
               "--working-directory=" + str(REPO)]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
                 "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONHASHSEED"):
        command += ["--setenv=" + name + "=" + env[name]]
    command += worker
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / (args.out.name + "-launch.json")).write_text(json.dumps(dict(unit=unit, command=command,
        synthetic_receipt=str(dry / "status.json"), approval="Pat: go for B1 and B2"), indent=2) + "\n")
    print(json.dumps(dict(starting_unit=unit, stage=args.stage, out=str(args.out))), flush=True)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
