#!/usr/bin/env python3
"""Committed execution wrapper for phase-b-rome-b1-b2-v1; no method changes.

Real execution requires the bounded launch_phase_b.py service. The synthetic
path exercises the SAME stage functions on a tiny CPU fixture. It never
opens the blind directory and cannot accept a test-set spec path.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools/edit_cert"))
from rome_solver.covariance import collect_model_covariances
from rome_solver.solver import Budget, CONTEXT_PREFIXES, RomeConfig, RomeSolver, request_from_spec

PREFLIGHT = REPO / "docs/research-data/phase-b0-preflight"
PROTOCOL = REPO / "docs/phase-b-b1-b2-protocol.md"
FACTS = tuple(sorted(json.loads((PREFLIGHT / "budget.json").read_text())["facts"]))
LAYERS = (4, 8, 12)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def verify_b0():
    manifest = json.loads((REPO / "docs/research-data/phase-b0-manifest.json").read_text())
    for name, receipt in manifest["files"].items():
        if sha(REPO / name) != receipt["sha256"]:
            raise ValueError(f"B0 hash drift: {name}")
    return manifest


def make_preflight(model_root, out):
    verify_b0()
    if out.exists():
        raise FileExistsError(out)
    root = model_root / "gemma-4-E2B-it"
    architecture = json.loads((PREFLIGHT / "architecture.json").read_text())
    for name, expected in architecture["metadata_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Model metadata drift: {name}")
    source = Path(importlib.util.find_spec("transformers").origin).parent / architecture["modeling_source"].removeprefix("transformers/")
    if sha(source) != architecture["modeling_source_sha256"]:
        raise ValueError("Installed modeling source drift")
    with (root / "model.safetensors").open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = f.read(size)
    if hashlib.sha256(header).hexdigest() != architecture["safetensors_header_sha256"]:
        raise ValueError("Checkpoint header drift")
    checkpoint = root / "model.safetensors"
    before = checkpoint.stat()
    full_hash = sha(checkpoint)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during hashing")
    sources = json.loads((PREFLIGHT / "covariance-sample.json").read_text())["sources"]
    datasets = Path("/mnt/models/distill-course/hf/datasets")
    for entry in sources:
        path = REPO / entry["path"] if entry["source"] == "docs" else datasets / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Corpus source drift: {entry['path']}")
    receipt = dict(protocol_sha256=sha(PROTOCOL), checkpoint_sha256=full_hash,
                   checkpoint_bytes=after.st_size, checkpoint_mtime_ns=after.st_mtime_ns,
                   model_root=str(model_root.resolve()), metadata_sha256=architecture["metadata_sha256"],
                   modeling_source_sha256=architecture["modeling_source_sha256"],
                   covariance_tokens_sha256=sha(PREFLIGHT / "covariance-tokens.npy"),
                   pretrained_model_forwards=0, source_files_verified=len(sources),
                   wrapper_sha256=sha(Path(__file__)), approval="Pat: go for B1 and B2")
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, receipt)
    return receipt


def verify_preflight(path, model_root):
    verify_b0()
    record = json.loads(path.read_text())
    if record["protocol_sha256"] != sha(PROTOCOL) or record["wrapper_sha256"] != sha(Path(__file__)):
        raise ValueError("Execution/protocol hash drift")
    if record["model_root"] != str(model_root.resolve()):
        raise ValueError("Model root differs from preflight")
    root = model_root / "gemma-4-E2B-it"
    st = (root / "model.safetensors").stat()
    if (st.st_size, st.st_mtime_ns) != (record["checkpoint_bytes"], record["checkpoint_mtime_ns"]):
        raise ValueError("Checkpoint stat changed since full hash")
    for name, expected in record["metadata_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Model metadata drift: {name}")
    source = Path(importlib.util.find_spec("transformers").origin).parent / "models/gemma4/modeling_gemma4.py"
    if sha(source) != record["modeling_source_sha256"]:
        raise ValueError("Modeling source drift")
    return record


def verify_service():
    if not os.environ.get("INVOCATION_ID"):
        raise RuntimeError("Real stage requires a systemd user service")
    lock = Path.home() / "Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock"
    with lock.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(f, fcntl.LOCK_UN)
            raise RuntimeError("Shared flock is not held by the waiting launcher")
    cg = next(line.split("::", 1)[1].strip() for line in Path("/proc/self/cgroup").read_text().splitlines() if "::" in line)
    root = Path("/sys/fs/cgroup") / cg.lstrip("/")
    expected = {"memory.high": 22 * 1024**3, "memory.max": 24 * 1024**3, "memory.swap.max": 8 * 1024**3}
    limits = {name: int((root / name).read_text().strip()) for name in expected}
    if any(limits[name] > cap for name, cap in expected.items()):
        raise RuntimeError("Cgroup memory limits exceed approved bounds")
    return dict(cgroup=cg, limits=limits, invocation_id=os.environ["INVOCATION_ID"])


def execute(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.stage == "b2" and (args.layer is None or args.fact_id is None or args.covariance is None):
        raise ValueError("B2 requires a registered fact, layer and B1 covariance directory")
    budget = Budget(max_forwards=200 if args.stage == "b1" else 130, max_gradient_steps=25)
    started = time.monotonic()
    torch.set_num_threads(4)
    record = None if args.synthetic else verify_preflight(args.preflight, args.model_root)
    limits = None if args.synthetic else verify_service()
    args.out.mkdir(parents=True)
    attempts, completed = [], []
    status = dict(status="starting", stage=args.stage, synthetic=args.synthetic,
                  fact_id=args.fact_id, layer=args.layer, protocol_sha256=sha(PROTOCOL),
                  wrapper_sha256=sha(Path(__file__)), limits=limits,
                  preflight_sha256=None if args.synthetic else sha(args.preflight),
                  checkpoint_sha256=None if record is None else record["checkpoint_sha256"],
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    write(args.out / "status.json", status)
    try:
        if args.synthetic:
            sys.path.insert(0, str(HERE / "tests"))
            from test_rome import Model, Tokenizer
            model = Model(depth=14).eval()
            layers, tokenizer = model.layers, Tokenizer()
            info = dict(synthetic=True)
        else:
            from edit_cert.instruments.nf4 import Nf4Instrument
            inst = Nf4Instrument.load("E2B", args.model_root)
            model, layers, tokenizer = inst.model, inst.layers, inst.tok
            info = inst.fingerprint()
        status.update(status="running", instrument=info)
        write(args.out / "status.json", status)

        def before(mod, inputs):
            attempts.append(budget.forwards)

        def after(mod, inputs, output):
            completed.append(budget.forwards)
            if len(completed) % 10 == 0 or len(completed) in (1, 130, 200):
                progress = dict(completed_forwards=len(completed), counted_forwards=budget.forwards,
                                optimizer_updates_started=budget.gradient_steps,
                                elapsed_seconds=time.monotonic() - started)
                write(args.out / "progress.json", progress)
                print(json.dumps(progress), flush=True)

        handles = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
        try:
            if args.stage == "b1":
                if args.synthetic:
                    windows = np.random.default_rng(20261026).integers(3, 128, size=(200, 512), dtype=np.int64)
                else:
                    windows = np.load(PREFLIGHT / "covariance-tokens.npy", allow_pickle=False)
                matrices, meta = collect_model_covariances(model, layers, windows, LAYERS,
                                                          synthetic=args.synthetic, budget=budget)
                for layer, matrix in matrices.items():
                    # Validate the SAVED float32 C as SPD before B1 can succeed.
                    torch.linalg.cholesky(torch.as_tensor(matrix, dtype=torch.float64))
                    np.save(args.out / f"C{layer}.npy", matrix, allow_pickle=False)
                status.update(covariance=meta,
                              covariance_sha256={str(l): sha(args.out / f"C{l}.npy") for l in LAYERS})
            else:
                covariance_status = json.loads((args.covariance / "status.json").read_text())
                if covariance_status["status"] != "complete" or covariance_status["synthetic"] != args.synthetic:
                    raise ValueError("B1 covariance incomplete or synthetic/real mismatch")
                if not args.synthetic and covariance_status["checkpoint_sha256"] != record["checkpoint_sha256"]:
                    raise ValueError("Covariance checkpoint differs")
                covariance_path = args.covariance / f"C{args.layer}.npy"
                if sha(covariance_path) != covariance_status["covariance_sha256"][str(args.layer)]:
                    raise ValueError("Covariance artifact hash mismatch")
                spec_path = REPO / f"tools/edit_cert/facts/phase-b-dev/{args.fact_id}.json"
                spec = json.loads(spec_path.read_text())
                solver = RomeSolver(model, layers, tokenizer, synthetic=args.synthetic, kv_deadline=13)
                solution = solver.solve(args.layer, request_from_spec(spec), CONTEXT_PREFIXES,
                                        np.load(covariance_path, allow_pickle=False), RomeConfig(), budget)
                solution.metadata.update(checkpoint_sha256=None if record is None else record["checkpoint_sha256"],
                                         protocol_sha256=sha(PROTOCOL), covariance_source_sha256=sha(covariance_path),
                                         dev_spec_sha256=sha(spec_path), execution_wrapper_sha256=sha(Path(__file__)))
                solution.save(args.out / "edit")
                status.update(edit_content_sha256=solution.backend().content_sha256(),
                              loss_first=solution.metadata["history"][0], loss_last=solution.metadata["history"][-1],
                              edit_manifest_sha256=sha(args.out / "edit/solver.json"))
        finally:
            for h in handles:
                h.remove()
        expected = 200 if args.stage == "b1" else 130
        if budget.forwards != expected or len(completed) != expected or budget.gradient_steps != (0 if args.stage == "b1" else 25):
            raise ValueError("Forward/update count differs from fixed protocol")
        status.update(status="complete", forwards=budget.forwards, completed_forwards=len(completed),
                      gradient_steps=budget.gradient_steps, elapsed_seconds=time.monotonic() - started)
        write(args.out / "status.json", status)
        return status
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      forwards_attempted=budget.forwards, completed_forwards=len(completed),
                      gradient_steps=budget.gradient_steps, elapsed_seconds=time.monotonic() - started)
        write(args.out / "status.json", status)
        (args.out / "failure.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "b1", "b2"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("/mnt/models/hf-models"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fact-id", choices=FACTS)
    parser.add_argument("--layer", type=int, choices=LAYERS)
    parser.add_argument("--covariance", type=Path)
    args = parser.parse_args()
    if args.stage == "preflight":
        result = make_preflight(args.model_root, args.out)
    else:
        if not args.synthetic and args.preflight is None:
            parser.error("Real execution requires --preflight")
        result = execute(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
