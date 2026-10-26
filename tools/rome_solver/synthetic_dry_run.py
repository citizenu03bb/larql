#!/usr/bin/env python3
"""B0 rehearsal with tiny CPU fixtures; NEVER loads pretrained weights.

Writes a synthetic edit with the exact registered optimizer settings, all
algebra witnesses, three covariance sites and a low_rank reload receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
from test_rome import Model, REQUEST, Tokenizer
from edit_cert import backends
from rome_solver.covariance import collect_model_covariances
from rome_solver.solver import CONTEXT_PREFIXES, RomeSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    model = Model(depth=14).eval()
    tokens = np.random.default_rng(20261026).integers(3, 128, size=(8, 32), dtype=np.int64)
    covariances, cov_meta = collect_model_covariances(model, model.layers, tokens, (4, 8, 12),
                                                    synthetic=True, positions=(7, 15, 23, 31))
    solver = RomeSolver(model, model.layers, Tokenizer(), synthetic=True)
    result = solver.solve(8, REQUEST, CONTEXT_PREFIXES, covariance=covariances[8])
    result.save(args.out / "edit")
    loaded = backends.load(args.out / "edit")
    if loaded.content_sha256() != result.backend().content_sha256():
        raise ValueError("Synthetic edit reload mismatch")
    receipt = dict(synthetic=True, pretrained_model_forwards=0, covariance=cov_meta,
                   solve_forwards=result.metadata["forwards"], optimizer_steps=result.metadata["gradient_steps"],
                   rank_one_error_maxabs=float(np.abs(result.w_k_star + result.u @ (result.v.T @ result.k_star) - result.v_star).max()),
                   roundtrip_content_sha256=loaded.content_sha256(),
                   saved_proof_files=["k_star.npy", "v_star.npy", "w_k_star.npy", "C.npy"],
                   solver_source_sha256=hashlib.sha256((HERE / "rome_solver/solver.py").read_bytes()).hexdigest(),
                   covariance_source_sha256=hashlib.sha256((HERE / "rome_solver/covariance.py").read_bytes()).hexdigest(),
                   fixture_source_sha256=hashlib.sha256((HERE / "tests/test_rome.py").read_bytes()).hexdigest(),
                   independence="same-author self-test; not a certification")
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
