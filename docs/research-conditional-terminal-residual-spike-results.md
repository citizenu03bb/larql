# Conditional terminal-residual spike result

**Run date:** 2026-08-25  
**Frozen source:** `ad5e2a50a1ae8d2487d4c609a24b45eeeb2cef79`  
**Outcome:** implementation-inconclusive before adapter compilation  
**Validation3:** sealed and not accessed

## Verdict

The one permitted E4B run did not test the terminal-residual hypothesis. The
three spent validation2 positives completed their contextual forwards, after
which the deterministic compiler stopped with:

```text
terminal adapter refuses final logit softcap 30
```

Gemma 4 applies `30 * tanh(logit / 30)` at the final vocabulary projection.
The frozen compiler implemented the final RMSNorm and vocabulary projection
but deliberately rejected architectures with a final-logit softcap. This
architectural precondition should have been detected from `index.json` before
the expensive run. No direction, alpha, or derived patch was produced.

This is an implementation/preflight failure, not evidence against the proposed
adapter. It also supplies no positive evidence for it.

## Protocol accounting

- K1 loaded and remained bound to the expected base checksum.
- All three synthesis forwards completed before the compiler refusal.
- No adapter SHA-256 was emitted.
- The validation3 loader remained behind the adapter-hash gate and was never
  reached.
- No derived `.vlp` exists.
- The cached-generation phase was not reached.
- The run was not repeated and no resource limit was changed.
- Alder artifacts and holdouts remained out of scope.

## Resource record

- systemd user unit: `larql-terminal-residual-20260825-ad5e2a50.service`
- exit: status 1, `Result=exit-code`
- wall time: 1 minute 49.253 seconds
- CPU time: 7 minutes 42.630 seconds
- memory peak: 23,622,582,272 bytes, reported by systemd as 22G
- swap peak: 4,593,311,744 bytes, reported by systemd as 4.2G
- frozen limits: `MemoryHigh=22G`, `MemoryMax=24G`,
  `MemorySwapMax=8G`, `RuntimeMaxSec=3600`

The resource envelope worked as intended: the process crossed the soft high
water mark, used bounded swap, and exited through its own fail-closed compiler
rather than an OOM kill.

## Evidence identities

- preregistration SHA-256:
  `c5a88be668f2bb5f952c49e00e7f1b8c679f4b4d49ff7f21545ca9100b65d687`
- pre-run manifest SHA-256:
  `5a7b658ce6314f5bbf58d00fdc2dceefb441553cf96317e384bce066db23e6e3`
- release binary SHA-256:
  `5a219dda8503fc2416da92822d1c69df271746c33843d492d00e9b344e9ff4e5`
- failure evidence SHA-256:
  `6034253cb8d5cca1121ea737e5ba364b1f8808eed1bf70a1b61a62f8f7e416e4`

The manifest, failure evidence, and journal are retained in the private
research directory beside the Galadriel patch.

## Consequence

Version 1 is closed as implementation-inconclusive and must not be repaired or
re-run. A future version is a new protocol. Before it may request E4B, it must:

1. implement the exact final-logit softcap in baseline scoring, runner
   selection, the analytic gradient, the monotonicity grid, and bisection;
2. add synthetic disagreement-point tests where the capped and uncapped
   runner/margin differ;
3. run a zero-weight preflight against the real `index.json` that exercises
   every compiler architecture precondition without loading E4B;
4. freeze a new preregistration and manifest with a distinct version and run
   identity.

Until those gates are complete, Alder-v2 remains blocked and no further
heavy-model run is justified.
