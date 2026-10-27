# P1-2 covariance complete — Codex, 2026-10-27

The shared matrix is published at
`~/Projects/larql-research/phase-b-p1-v2-20261027/covariance/C17.npy`, with
its atomic completion receipt in `covariance/status.json`.

- float32, **6400 × 6400**, input to `transformer.h.17.mlp.c_proj`;
- SHA-256 **`a9c6faa06c9b9155e764a9a38298c62c0a11379369bb3869075a76f61057c715`**;
- **1,228,800 keys**, every position of 1200 × 1024 fixed WikiText-103
  windows, seed 20261026; 192 keys per dimension;
- float64 uncentred second moment, one final ridge of 1e-4 times mean
  diagonal, added value **2.7694381490532944e-6**; SPD validation passed.

The initial A1 400-window service reached its internal time deadline after
**272 forwards** and stopped. Its logs and failed receipt are preserved;
no matrix from that attempt was published or included in the final moment.
Pat approved A2 before any A2 model service: six parts of 200, using the
same frozen windows, estimator, bounds and single final ridge. All six
completed, totaling **1200 successful forwards**. **Actual P1-2 use is
1472 forwards and zero optimizer updates**, below A2's explicit 1600 bound.
There was no retry of an A2 failure, no resampling and no change to the
model or estimator. Every model service used the shared flock and bounded
user systemd execution, after a fresh synthetic rehearsal.

The local model and both parquet files were checksum-verified before the
first forward. The 29,444 article records and complete window metadata
were frozen before inference. Window file SHA-256:
`f6bb7e1c986ffe343e47610ffd2a94820af379d81625873d357455498b6d9de9`.
The matrix's hash, shape and dtype were independently read back after
publication. The [compact result receipt](research-data/phase-b-p1-covariance-results-codex.json)
pins all six part hashes and counters; raw sums and sample files remain
in the shared covariance directory for later model-free auditing.

This is a fixed prior for the two independent solves, not an edited fact
or a certification. The next authorized order is Claude's five solves,
then Codex's five. Before P1-4, Codex reads only
`solves-claude/completion.json` from the peer outputs, and no peer solve
source or factors. No push or blind-set access was performed.
