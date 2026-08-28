# Galadriel compiled-surface C1 result

**Run date:** 2026-08-28

**Frozen source:** `b657cd67c1b152c1794e47852f0887d23c4b3501`

**Outcome:** passed (`7/7` positives, `21/21` controls)

**Claim boundary:** product-only result on spent Galadriel development
material

## Verdict

C1 passed all 28 frozen contracts. Exact compiled prompt-surface eligibility
can replace the learned residual detector for the enumerated Galadriel
surfaces when it gates the complete frozen effect: the L31/F1625 FFN override
row plus the version-2 terminal residual adapter.

This is the selected product architecture. It does not discover paraphrases,
infer aliases, detect semantic relations, certify an FPR, or establish a new
fact. Unsupported surfaces are rejected by construction.

## Positive contracts

All seven frozen supported prompts:

- returned exactly one `fire` decision with the expected suffix index;
- emitted exactly `[8833, 553, 21457]`;
- triggered logical slot L31/F1625 once;
- released `ActiveSpan` immediately after the third target ID.

The seven prompts cover the canonical and stored paraphrases from the spent
validation2 and validation3 suites. They are enumerated product coverage, not
held-out linguistic generalization.

## Control contracts

All twenty-one factorial controls:

- returned `compiled_reject`;
- exposed neither the FFN row nor the terminal residual delta;
- never triggered or armed `ActiveSpan`;
- produced complete three-token outputs bit-identical to an independent
  source-Q4K generation.

This proves fail-closed implementation for the enumerated evaluation set. It
does not turn zero observed failures on spent controls into a statistical FPR
bound.

## What C0 and C1 establish together

The controlled architectural contrast is decisive on this artifact:

| Candidate | Positive contracts | Control contracts | Verdict |
|---|---:|---:|---|
| C0: compiled gate + terminal adapter only | 1/7 | 21/21 | failed |
| C1: compiled gate + FFN row + terminal adapter | 7/7 | 21/21 | passed |

The terminal adapter is therefore a complement to the FFN payload, not a
standalone replacement. Eligibility and effect are still cleanly separated:
the exact matcher alone decides visibility; FFN activation is never consulted
as a gate. `ActiveSpan` authorization comes from that same explicit decision.

## Product contract

The honest deliverable is:

```text
exact bound artifact
+ exact enumerated prompt suffix
+ frozen composite effect
-> exact first target ID + ActiveSpan continuation

anything else
-> compiled_reject / exact base path
```

Before any future runtime or wire integration, the compiler/runtime must retain
the current equality checks across base checksum, slot, target IDs, adapter
application, every delta value, calibration metadata, and live FFN Insert.
The learned detector vectors in the source patch are not part of product
eligibility and should not be copied into a future product wire representation.

## Evidence identities and resources

- C1 preregistration SHA-256:
  `01bf8a82bb26cebdbf83c73325ad4f5b8ab8a1864c610ffdf175568e94cd4fdb`
- C1 pre-run manifest SHA-256:
  `32f88e7f20812cac52fa6101e387c8d6637d3a206fe9960171a5a4748352e47a`
- compiled surface manifest SHA-256:
  `0aa1c4f29004f6efe8273c6566333c55d0b5c82e24488de3a6ec2006f2329cb3`
- effect patch SHA-256:
  `37c4088ab1a9cd4f8452b8a59956a990542d4f8fac1f3728d2b6e8a35e84e9a6`
- release binary SHA-256:
  `56671117e9d25301f196484f67d75b938e852cf006b769428ad56bd828e639f3`
- result evidence SHA-256:
  `38ae90a21deded36acecba6d7077a0e6e951d06ad785497b71636f4745b6d40b`
- systemd unit:
  `larql-product-surface-c1-20260828-b657cd67.service`
- invocation ID:
  `17edee982fab4d5d8c215c1e6e7a2cfc`
- wall time: 31 minutes 4.514 seconds
- CPU time: 3 hours 50 minutes 28.908 seconds
- memory peak: 22G
- swap peak: 5.9G

