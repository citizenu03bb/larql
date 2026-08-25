#!/usr/bin/env python3
"""Attach one frozen, runtime-compatible detector to one compiled payload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_f32(values: list[float]) -> str:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("detector vectors must be non-empty and finite")
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path)
    parser.add_argument("detector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--feature", type=int)
    parser.add_argument("--target-ids", type=int, nargs="+", required=True)
    parser.add_argument("--vindex", type=Path, required=True)
    parser.add_argument("--protocol", default="contextual-detector-attachment")
    args = parser.parse_args()

    target_ids = args.target_ids
    if len(target_ids) < 2 or any(value < 0 for value in target_ids):
        raise ValueError("--target-ids must contain at least two non-negative IDs")
    patch = json.loads(args.patch.read_text())
    detector = json.loads(args.detector.read_text())
    config = json.loads((args.vindex / "index.json").read_text())
    checksums = config.get("checksums") or {}
    if not checksums:
        raise ValueError("base vindex has no checksum manifest")
    digest = hashlib.sha256()
    for filename, checksum in sorted(checksums.items()):
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(checksum.encode())
        digest.update(b"\n")
    patch["base_checksum"] = f"sha256-manifest:{digest.hexdigest()}"
    inserts = [operation for operation in patch["operations"] if operation["op"] == "insert"]
    if (args.layer is None) != (args.feature is None):
        raise ValueError("--layer and --feature must be provided together")
    if args.layer is None:
        if len(inserts) != 1:
            raise ValueError(f"patch must contain exactly one Insert, found {len(inserts)}")
        insert = inserts[0]
    else:
        matching = [
            operation
            for operation in inserts
            if int(operation["layer"]) == args.layer
            and int(operation["feature"]) == args.feature
        ]
        if len(matching) != 1:
            raise ValueError("requested payload slot must identify exactly one Insert")
        insert = matching[0]
    down_meta = insert.get("down_meta")
    if down_meta is not None and target_ids[0] != int(down_meta["i"]):
        raise ValueError("first target ID does not match the Insert down metadata")
    if detector.get("representation") != "mean_no_bos":
        raise ValueError("runtime supports only mean_no_bos detectors")
    if detector.get("method") not in {
        "raw_conjunction",
        "unit_conjunction",
        "unit_factor_prototypes",
    }:
        raise ValueError("detector method is not representable by the runtime ABI")
    required_vectors = ("center", "entity_direction", "relation_direction")
    vectors = {name: detector[name] for name in required_vectors}
    widths = {len(values) for values in vectors.values()}
    if len(widths) != 1:
        raise ValueError("detector vectors have inconsistent widths")
    detector_layer = int(detector["layer"])
    if detector_layer >= int(insert["layer"]):
        raise ValueError("detector layer must precede the payload layer")
    thresholds = {
        name: float(detector[name])
        for name in ("entity_threshold", "relation_threshold")
    }
    if any(not math.isfinite(value) for value in thresholds.values()):
        raise ValueError("detector thresholds must be finite")

    calibration = json.dumps(
        {
            "protocol": args.protocol,
            "detector_sha256": sha256(args.detector),
            "selection_protocol": detector.get("selection_protocol"),
            "selection_metrics": detector.get("selection_metrics"),
            "method": detector["method"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    patch["operations"] = [
        operation
        for operation in patch["operations"]
        if not (
            operation.get("op") == "context"
            and operation.get("layer") == insert["layer"]
            and operation.get("feature") == insert["feature"]
        )
    ]
    patch["operations"].append(
        {
            "op": "context",
            "layer": insert["layer"],
            "feature": insert["feature"],
            "target_ids": target_ids,
            "detector": {
                "detector_layer": detector_layer,
                "representation": detector["representation"],
                "center_vector_b64": encode_f32(vectors["center"]),
                "entity_direction_b64": encode_f32(vectors["entity_direction"]),
                "relation_direction_b64": encode_f32(vectors["relation_direction"]),
                "entity_threshold": thresholds["entity_threshold"],
                "relation_threshold": thresholds["relation_threshold"],
                "calibration": calibration,
            },
            "residual_adapter": None,
        }
    )
    patch["description"] = (
        "Alder-v2 development patch: frozen detector-gated payload before "
        "terminal residual compilation"
    )
    for tag in ("research-spike", "contextual", "alder-v2"):
        if tag not in patch.setdefault("tags", []):
            patch["tags"].append(tag)
    atomic_json(args.output, patch)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "slot": [insert["layer"], insert["feature"]],
                "detector_layer": detector_layer,
                "target_ids": target_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
