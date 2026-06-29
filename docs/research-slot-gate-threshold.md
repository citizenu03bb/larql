# Research Branch: Insert Slot Gate Threshold

This branch is a research branch for training-free, vindex-level fact insertion.
It is intended to be shared for review and reproduction, not presented as a
finished product surface.

Base:

- Rebases cleanly on upstream `origin/main` at `b6d5e8d5`.
- Keeps upstream's current vindex split and Q4K layout.
- The small upstreamable fixes have been extracted into separate PRs.

## What This Branch Adds

- Per-slot gate thresholds for inserted feature overrides.
- Python bindings for low-level edit primitives used by the experiments:
  `set_up_vector`, `set_down_vector`, `set_gate_vector`, and
  `set_gate_threshold`.
- Q4K-aware Python `Vindex.infer()` dispatch through `InferenceWeights`.
- A small HTTP daemon for live edit experiments:
  `experiments/05_phase3_agent/larql_daemon.py`.
- Smoke scripts for single-fact edit validation:
  `experiments/05_phase3_agent/smoke_tau.sh` and
  `experiments/05_phase3_agent/larql_agent_demo.sh`.

The core edit recipe is:

- Capture target and contrast residuals with `infer_trace`.
- Build a gate vector from the target residual projected away from contrast
  residuals.
- Set the inserted slot's up vector parallel to the gate vector.
- Set the down vector toward the target token embedding.
- Calibrate a per-slot threshold `tau` so the target prompt still pushes while
  weak off-target activations are suppressed.

## Verified Locally

The rebased branch was checked with:

```bash
cargo check -p larql-vindex -p larql-inference -p larql-python
cargo test -p larql-vindex -p larql-inference -p larql-python --lib
uv run --no-sync maturin develop --release
```

The live daemon smoke was also run against the default E4B Q4K vindex path:

```bash
crates/larql-python/.venv/bin/python experiments/05_phase3_agent/larql_daemon.py --port 8770
LARQL_PORT=8770 bash experiments/05_phase3_agent/smoke_tau.sh
```

Observed smoke result:

- `/edit` auto-calibrated `tau` to `1.5`.
- `The capital of Atlantis is` pushed to `Pose` at about 100%.
- Belgium, Spain, and Egypt stayed on native answers.
- `/reset` restored the Atlantis baseline.

## How To Reproduce

Build and install the Python extension:

```bash
cargo check -p larql-python
uv run --no-sync maturin develop --release
```

Start the daemon. The default vindex path is
`/home/pat/larql-vindexes/gemma-4-e4b.vindex`; override it with `--vindex` if
needed.

```bash
crates/larql-python/.venv/bin/python experiments/05_phase3_agent/larql_daemon.py \
  --port 8770 \
  --vindex /path/to/gemma-4-e4b.vindex
```

Run the fast single-fact smoke:

```bash
LARQL_PORT=8770 bash experiments/05_phase3_agent/smoke_tau.sh
```

Or run the agent-facing demo:

```bash
LARQL_PORT=8770 bash experiments/05_phase3_agent/larql_agent_demo.sh
```

## Known Limits

- This is validated as a single-token or first-subtoken push mechanism. It does
  not yet solve robust multi-token target insertion.
- The daemon is an experiment harness, not a production API. It keeps edit
  overlays in process memory and resets them with `/reset`.
- The method depends on prompt form, model family, layer choice, contrast set,
  and calibration grid.
- Sequential multi-fact insertion has improved slot allocation, but broad
  scaling/generalization is still research work.
- The public branch should not be treated as upstream-ready in full. The
  upstream-ready pieces are the smaller extracted PRs.

## Related Docs

- `docs/training-free-insert.md`: original training-free insertion write-up.
- `experiments/05_phase3_agent/`: daemon and agent-facing smoke scripts.
- `experiments/06_scaling/`: scaling and multi-token exploratory scripts.
