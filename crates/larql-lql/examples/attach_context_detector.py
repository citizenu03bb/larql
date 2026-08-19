#!/usr/bin/env python3
"""Append a frozen contextual detector to one payload in a .vlp patch."""

from __future__ import annotations

import argparse
import base64
import json
import struct
from pathlib import Path
import hashlib


def encode_f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path)
    parser.add_argument("detector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--feature", type=int)
    parser.add_argument(
        "--target-ids",
        type=int,
        nargs="+",
        required=True,
        help="complete tokenizer IDs, including the naturally selected first token",
    )
    parser.add_argument(
        "--vindex",
        type=Path,
        required=True,
        help="exact base vindex whose index.json checksum manifest binds the detector",
    )
    args = parser.parse_args()

    patch = json.loads(args.patch.read_text())
    frozen = json.loads(args.detector.read_text())
    config = json.loads((args.vindex / "index.json").read_text())
    checksums = config.get("checksums") or {}
    if not checksums:
        parser.error("base vindex has no checksum manifest")
    digest = hashlib.sha256()
    for filename, checksum in sorted(checksums.items()):
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(checksum.encode())
        digest.update(b"\n")
    patch["base_checksum"] = f"sha256-manifest:{digest.hexdigest()}"
    inserts = [op for op in patch["operations"] if op.get("op") == "insert"]
    if (args.layer is None) != (args.feature is None):
        parser.error("--layer and --feature must be provided together")
    if args.layer is None:
        if len(inserts) != 1:
            parser.error(f"patch must contain exactly one insert, found {len(inserts)}")
        payload_layer = int(inserts[0]["layer"])
        payload_feature = int(inserts[0]["feature"])
    else:
        payload_layer = args.layer
        payload_feature = args.feature

    if len(args.target_ids) < 2:
        parser.error("--target-ids requires a multi-token target (at least two IDs)")
    payload_inserts = [
        op
        for op in inserts
        if int(op["layer"]) == payload_layer and int(op["feature"]) == payload_feature
    ]
    if len(payload_inserts) != 1:
        parser.error(
            f"payload L{payload_layer}/F{payload_feature} must identify exactly one insert"
        )
    down_meta = payload_inserts[0].get("down_meta")
    if down_meta is not None and args.target_ids[0] != int(down_meta["i"]):
        parser.error(
            f"first target ID {args.target_ids[0]} does not match insert first ID {down_meta['i']}"
        )

    detector_layer = int(frozen["layer"])
    if detector_layer >= payload_layer:
        parser.error(
            f"detector L{detector_layer} must precede payload L{payload_layer}"
        )
    dimensions = {
        len(frozen["center"]),
        len(frozen["entity_direction"]),
        len(frozen["relation_direction"]),
    }
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        parser.error("detector vectors must have one equal, non-zero dimension")

    patch["operations"] = [
        op
        for op in patch["operations"]
        if not (
            op.get("op") == "context"
            and op.get("layer") == payload_layer
            and op.get("feature") == payload_feature
        )
    ]
    patch["operations"].append(
        {
            "op": "context",
            "layer": payload_layer,
            "feature": payload_feature,
            "target_ids": args.target_ids,
            "detector": {
                "detector_layer": detector_layer,
                "representation": frozen["representation"],
                "center_vector_b64": encode_f32(frozen["center"]),
                "entity_direction_b64": encode_f32(frozen["entity_direction"]),
                "relation_direction_b64": encode_f32(frozen["relation_direction"]),
                "entity_threshold": frozen["entity_threshold"],
                "relation_threshold": frozen["relation_threshold"],
                "calibration": frozen.get("selection_protocol"),
            },
        }
    )
    args.output.write_text(json.dumps(patch, indent=2) + "\n")


if __name__ == "__main__":
    main()
