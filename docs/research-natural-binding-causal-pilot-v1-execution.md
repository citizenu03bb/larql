# Natural Binding Causal Pilot v1 — execution instrument

The user approved the model-free protocol on 2026-09-24. The scientific
configuration, corpus, thresholds and 368-forward maximum remain exactly as
approved in design commit `7c307eda11b50fcb74ebdcb3592bb270e3acb3a8`.

## Instrument

- `natural_binding_causal_pilot.rs` is a bounded native-Q4K worker. It performs
  metadata-only preflight with the same architecture-aware encoding path as
  LQL's crate-private `encode_vindex_prompt`, checks all frozen Python token
  IDs in Rust, and uses the unpatched `VectorIndex` directly. No overlay is
  instantiated. `hidden_to_raw_logits` already applies the configured softcap.
- The worker only accepts the stage's exact 184-job sequence. Model prompts
  come from the frozen corpus, not from requests. It rejects wrong IDs, arms,
  ordinals, extra forwards and malformed/nonfinite replacement rows.
- `run_natural_binding_pilot.py` verifies every frozen source/model/runtime
  hash before worker startup. It persists and fsyncs each attempted forward
  before requesting it, then saves the full f32 logits and relevant residual
  or exact f32 replacement as a compressed NPZ before journaling completion.
- The controller uses the unchanged numerical reference to fit the direction
  and generate all nine arms. Float64 computation followed by one f32 cast is
  checked against the worker's actual row. The existing `PreFfnPatchHook`
  records the exact applied coordinate; synthetic Rust tests exercise that
  real hook without weights, including shape/position/finite rejection.
- Stage directories are exclusively created and can never be restarted. A
  crash consumes attempted forwards and cannot reset the ledger.
- Confirmation starts only through the controller after a discovery pass,
  exact 184-forward count, unchanged manifest, direction and evidence hashes,
  and re-evaluation of the discovery gates. Rejection happens before weight
  loading. The worker is an internal subprocess, not a standalone stage launcher.

## Order and stop accounting

The approved ledger interleaves each baseline with its disabled-hook replay.
The initial phase therefore costs 40 forwards for 20 prompts, as designed.
All 16 factual baseline answers are checked at the end of that phase. A failure
ends the stage as `eligibility_stop`; the direction is not fit and no
intervention is run. This is not a causal negative.

After eligibility passes, complete each nine-arm recipient row in ledger
order. Stop at the end of the first row that fails the frozen causal gate;
no subsequent recipient or confirmation forward is run. Mechanical drift,
nonfinite values, malformed accounting and infrastructure failures stop the
run as invalid rather than as scientific negatives. A numerically degenerate
fit stops before interventions. Only a complete 184-forward discovery pass
allows confirmation; only a complete confirmation pass supports the bounded
positive claim.

Execution uses the existing release target cache. Full local vindex files are
hashed with streaming 1 MiB buffers before launch; no tensor is loaded by the
freeze script. Each stage runs inside a unique `systemd --user` unit with
`MemoryHigh=22G`, `MemoryMax=24G`, `MemorySwapMax=8G` and
`RuntimeMaxSec=90min`. Check real processes before launch and after completion.

## Model-free validation

Run from the isolated branch:

```bash
python3 -m unittest discover -s scripts -p 'test_*natural_binding_pilot.py' -v
cargo test --release --locked -p larql-lql --example natural_binding_causal_pilot --target-dir /home/pat/Projects/larql.worktree/multitoken-span-emitter-v2/target
cargo build --release --locked -p larql-lql --example natural_binding_causal_pilot --target-dir /home/pat/Projects/larql.worktree/multitoken-span-emitter-v2/target
```

Then commit the instrument, freeze a new execution manifest, and run controller
`--preflight-only` for discovery. That path must perform zero model forwards.
The execution manifest freezes binary, protocol, approved dataset, tokenization,
ledger, controller, reference, analyzer/tests, Python binary, NumPy version and
all vindex files. A freeze manifest is separate from the immutable design
snapshot; it is not retroactively inserted into the original protocol.

Pre-run verification on 2026-09-24: 16 Python tests and 3 Rust tests passed.
The Python suite exercises complete successful and eligibility-failing controller
lifecycles using a fake worker, including rejection of corrupted evidence and
stage restarts. Rust preflight matched all 40 Python prompt/answer encodings,
confirmed the recorded base binding, and performed zero model forwards.
