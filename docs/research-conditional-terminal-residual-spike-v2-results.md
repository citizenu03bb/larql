# Conditional terminal-residual spike v2 result

**Run date:** 2026-08-25  
**Frozen source:** `6200ce92bb0d13c79b45b675e2cbba1c7c1583b0`  
**Outcome:** passed  
**Evaluation material:** spent Galadriel development suite, not a holdout

## Verdict

The one permitted E4B run passed every frozen version 2 gate. The deterministic
compiler produced and hashed a detector-gated terminal residual adapter, the
activation phase passed all four positives and twelve controls, and cached
generation passed the same sixteen contracts.

This establishes the proposed runtime seam as a viable capacity extension on
the Galadriel development fact. It does not certify a second fact, estimate
generalization, or change the failed Alder version 1 result.

## Compiler result

The compiler used only the three declared validation2 positives. On its worst
prompt, the exact softcapped target margin moved from `-0.133890` to
`+0.750084`. The frozen scalar solver selected:

- alpha: `0.951613`, below the fixed maximum `1.421908`;
- residual delta norm: `0.951612`;
- retained orthogonal-gradient ratio: `0.621884`, above the `0.10` gate;
- derived patch SHA-256:
  `37c4088ab1a9cd4f8452b8a59956a990542d4f8fac1f3728d2b6e8a35e84e9a6`.

The three compiled target margins were `2.109802`, `2.321205`, and
`0.750084`. These are in-sample compiler diagnostics, not evaluation evidence.

## Activation result

- Four of four detector-allowed positives selected target ID 8833.
- Their minimum target-versus-runner margin was `+2.417332`, above the frozen
  `+0.5` gate.
- Twelve of twelve controls remained detector-rejected.
- Every rejected control retained a complete f32 logit vector bit-identical to
  the K=1 reference path.
- Detector decisions and margins were bit-identical between reference and
  adapter paths.
- The largest observed residual-delta fraction was `0.033586`, below the fixed
  five-percent limit.

Thus all sixteen activation contracts passed.

## Cached-generation result

- Four of four positives emitted the exact first three IDs
  `[8833, 553, 21457]`, armed once, and released after the span.
- Twelve of twelve controls remained rejected and never armed ActiveSpan.
- All sixteen cached-generation contracts passed.

The terminal residual adapter is responsible only for strengthening the first
target token at the final prefill projection. The existing ActiveSpan state
machine still enforces the following two target IDs. This experiment does not
turn those continuation tokens into independently generated evidence.

## Protocol accounting

- The mandatory zero-weight preflight reported the exact E4B configuration as
  compatible and `weights_loaded=false`.
- The source, release binary, preflight, K=1 patch, vindex manifest, and both
  suites were hashed before launch.
- Validation3 was accessed only after the derived patch had been serialized,
  reparsed, checksum-checked, and hashed.
- One E4B process loaded the model once in the unique frozen systemd unit.
- No threshold, prompt, candidate, formula, resource limit, or output path was
  changed after manifest freeze.
- The run was not repeated.
- No Alder artifact or holdout was accessed.

## Resource record

- systemd user unit:
  `larql-terminal-residual-v2-20260825-6200ce92.service`
- exit: status 0, `Result=success`
- wall time: 16 minutes 39.387 seconds
- CPU time: 2 hours 34 minutes 35.220 seconds
- memory peak: 23,622,590,464 bytes, reported by systemd as 22G
- swap peak: 4,534,833,152 bytes, reported by systemd as 4.2G
- frozen limits: `MemoryHigh=22G`, `MemoryMax=24G`,
  `MemorySwapMax=8G`, `RuntimeMaxSec=3600`

The cgroup envelope again worked as intended. The model crossed the soft high
water mark and used bounded swap without approaching either hard limit or
incurring an OOM.

## Evidence identities

- preregistration SHA-256:
  `e8018970b4242bf986f46a4d0ed7d674cb59d5c7598137597e3d64df62c35ba9`
- pre-run manifest SHA-256:
  `9344a4807ee138747f1faba5b44a6a9c8d23b7dcb63ec6238db4e57359e93125`
- release binary SHA-256:
  `f5225e94db0a8120bb3e1cd6d63bc64b749cd418299fe66dd81d671a3e503b38`
- derived patch SHA-256:
  `37c4088ab1a9cd4f8452b8a59956a990542d4f8fac1f3728d2b6e8a35e84e9a6`
- evidence SHA-256:
  `73e1c587822bdfa081a23d48c50590a2add893bf2d536a71b16d147e92953534`
- systemd journal SHA-256:
  `2cb3a8743109917bce1ee38ef56b8797175e4f94d70ee15b9e8684fc132a1fcf`

The manifest, derived patch, full evidence, complete journal, and provenance
record are retained in the private research directory.

## Consequence

Version 2 closes as a mechanism-development pass. It supplies the missing
engineering basis for an Alder-v2 payload experiment, but it does not itself
authorize an Alder holdout open.

The next scientific protocol must remain separate from this Galadriel run. It
must freeze its payload mechanism, add the prospective passive and agentless
development stress suite, collect fresh residual captures after the new
payload is compiled, and satisfy the preregistered payload and LOSO detector
gates before the still-sealed Alder holdout can be opened once.
