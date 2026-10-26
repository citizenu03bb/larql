# Phase B2 certification protocol (Claude Code), frozen before any certification forward

**Mandate.** This protocol certifies Codex's 15 development edits independently: five public dev facts × L4/L8/L12, ROME rank 1 on E2B NF4. Those verdicts decide the B2 layer selection, under Codex's approved protocol `phase-b-rome-b1-b2-v1` (`docs/phase-b-b1-b2-protocol.md` on `research/phase-b-rome`; Pat's "go for B1 and B2"). Development results select a method. They are never reported as test results. The blinded test set is not opened.

**Instrument and edits.**
- Gemma 4 E2B on the NF4 bench (`nf4-e2b`), K/V write deadline block 13.
- Each edit is the `low_rank` adapter that Codex saved: `model.language_model.layers.L.mlp.down_proj`, with U0 (1536×1) and V0 (6144×1) in float32, applied at every token.
- The harness receives a minimal copy (backend.json, U0, V0), and its loader checks the content hash.
- The public spec is `tools/edit_cert/facts/phase-b-dev/<fact>.json`, and its SHA-256 must equal the handoff's.

**Before each certification (model-free).**
- `scripts/xcheck_rome_algebra.py`, written by Claude Code without importing Codex's solver, recomputes the following from the saved witnesses:
  - k\* as the float64 mean of the context keys;
  - q = C⁻¹k\*, U = v\* − Wk\* and V = q / (k\*ᵀq), compared with U0 and V0;
  - W′k\* = v\*.

  The tolerance is rtol/atol 2e-5. The check also compares every artifact hash, C against the B1 covariance of that layer, the content hash and the write site.
- A failure is recorded and the case is not certified. Nothing is repaired or rescaled.
- This check covers the rank-1 algebra only. It does not replicate the value optimisation.

**Certification, unchanged harness rules.**
- `prepare --instrument nf4-e2b --cap 400`, which gives four services: 312 native, 400 + 24 edit, 96 reload.
- Every mandatory control applies (C0–C4, C6, C7), with C5 and C8 reported:
  - native qualification, EOS, and truncation counted as unreadable;
  - efficacy in both subject orders;
  - attributes in both orders, and the answer-only mention with its rejected values;
  - the random rank-1 counterpart with equal column norms (seed from the spec);
  - off-target questions;
  - reload from the harness's own saved copy, in a later service.
- Greedy decoding, 24 new tokens at most, one trial.

**Execution.**
- `scripts/phase_b2_certify.py --run --i-have-approval` processes the solves listed `complete` in Codex's inventory, in its order, and waits for the pending ones.
- Each service goes through `run_service.sh`, which:
  - refuses to run unless this protocol is committed and unchanged;
  - holds the shared heavy-model `flock`, so Codex's solves and these services never overlap;
  - runs inside `systemd-run --user --wait` with MemoryHigh 22G, MemoryMax 24G, MemorySwapMax 8G and RuntimeMaxSec 1800;
  - sets CUBLAS_WORKSPACE_CONFIG `:4096:8`, offline flags, 4 CPU threads and PYTHONHASHSEED 0.

**Budget.**
- At most 400 forwards per service, under Q1, and no gradient steps.
- 15 × 832 = 12 480 forwards at most, in 60 services. Actual counts are recorded.
- Generations stop at EOS and are never extended.

**Stop rules.**
- Any failed service stops the whole series: OOM, nonfinite logits, cap or time exhausted, hash drift, instrument failure. There is no retry and no parameter change.
- One implementation repair may be proposed, in a new committed amendment.
- A solve that Codex marks `failed`, or one that fails the algebra check, is reported, not replaced.
- An 8-hour wall-clock limit applies to the driver.

**Rehearsal (model-free, required before the run).** `--rehearse` checks:
- the algebra of every complete solve;
- a minimal copy and a real plan for each, which must pack as [312, 400, 24, 96];
- that `run_service.sh` refuses to run without approval;
- the synthetic self-test at the default cap.

A separate CPU check applies a real U0/V0 to a synthetic module at the real path and compares it with the dense update.

**Prediction.** The value-optimisation losses that Codex recorded for the first fact stay high: NLL 25.1 → 20.6 at L4, which means the target stays improbable on the request prompt. Expected result: most dev edits refused on C1, with the edit unable to state the new value. This is a prediction, not a reading rule.

**Readings, per case.**
- `CERTIFIED`: a dev candidate passes every mandatory control, on development only.
- `REFUSED` with C1 FAIL: no efficacy.
- C1 PASS with C2 or C3 FAIL: efficacy that leaks into the entity's other attributes, the Phase 3 failure mode.
- NOT_EVALUATED reasons, such as a wrong native answer or a truncation, are reported as such and never counted as passes.

Codex computes the layer score afterwards from these verdicts: the number of facts with C1 PASS and C2 PASS. This protocol does not score layers.

**Outputs.**
- `~/Projects/larql-research/phase-b-certification-20261026/index.json`, with one entry per case:
  - verdict, every control status and the refusal reasons;
  - forwards per service;
  - hashes of the verdict, the evidence and the service records.
- Per-case run directories, holding the raw records with every generated id and text, and the boundary diagnostics; the driver log; a report; and an archive with a SHA-256 manifest.
- Codex then cross-checks the evidence model-free.
