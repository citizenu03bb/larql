# Execution wrapper for the approved B1–B2 protocol

Pat's instruction, "go for B1 and B2", authorizes the committed
`phase-b-rome-b1-b2-v1` protocol. This adds execution plumbing only;
the B0 method, sample, settings, protocol and certification rules remain
unchanged. No service was run before this wrapper was committed.

`tools/rome_solver/run_phase_b.py` provides three fixed operations:

- `preflight`: verify all B0 hashes, model metadata/source, frozen corpus
  sources and checkpoint header; stream a full checkpoint SHA-256, recording
  file size and modification time before/after. No model is loaded.
- `b1`: one service, the registered 200×512 windows and all three sites.
  Save C4/C8/C12, metadata and hashes. Validate each saved float32 C as SPD.
- `b2`: one of the five registered public dev facts, at L4/L8/L12, using
  complete matching B1 artifacts. Call the frozen solve, save every witness
  and provenance, require exactly 130 forwards and 25 optimizer updates.

Both real stages require a systemd user service, an already-held shared
flock, checked memory limits, a matching preflight receipt and a new output
directory. Progress tracks attempted and completed forwards, elapsed time
and started updates. Errors save failed status and traceback; a failed
service is never automatically retried. No unregistered fact path is accepted.

`launch_phase_b.py` verifies committed, unchanged code and the explicit
approval flag, runs a fresh synthetic instance of the same stage wrapper,
then execs `flock -w 3600 ... systemd-run --user --wait --pipe --collect`.
It applies 22G/24G/8G, 1800 seconds and a unique unit name. CPU threads are
4; TF32/deterministic behavior follows B0, with CUBLAS_WORKSPACE_CONFIG set
before startup. Hub/transformers offline flags prohibit downloads.

The synthetic B1 rehearsal uses 200×512 tiny-fixture tokens and the same
capture positions/counters. The synthetic B2 rehearsal first makes its own
synthetic covariances, then exercises the selected registered fact/layer
with the same optimizer settings and export. Synthetic and real covariance
receipts cannot be mixed. All synthetic artifacts are labelled.

Real artifacts go outside the worktree under
`larql-research/phase-b-rome-20261026/`. Only compact receipts/reports are
later versioned. The blind test directory remains unopened. Codex supplies
the 15 primary edits; Claude Code alone performs their certifications.
Layer selection and the gated identity ablation wait for those independent
verdicts, without Codex certifying its own method.
