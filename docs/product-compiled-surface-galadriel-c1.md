# Galadriel compiled-surface C1 evaluation

**Status:** frozen before C1 commit and before any C1 model load

**Protocol ID:** `compiled-surface-context-galadriel-product-v1-c1`

**Claim label:** `product-bounded`

## Change from closed C0

C0 proved that the terminal residual adapter alone was not the complete
effecter: it passed one of seven supported positives while all twenty-one
rejects stayed bit-identical to base. C1 changes exactly one architectural
factor. A compiled `fire` exposes the already-frozen L31/F1625 FFN row and
applies the already-frozen terminal residual delta. The learned detector stored
in the effect patch is not evaluated.

This is a new product architecture, not a retry or reinterpretation of C0. It
uses the same frozen surface bank, adapter, targets, and spent suites so the
comparison isolates the missing FFN effect row. It makes no prospective,
semantic-relation, or population-FPR claim.

## Frozen chain

```text
exact supported prompt suffix
    -> fire
    -> expose frozen L31/F1625 gate/up/down overrides
    -> apply frozen prefill-last-position residual delta
    -> natural first target ID 8833
    -> explicitly authorized ActiveSpan [553, 21457]

all other suffixes
    -> compiled_reject
    -> source Q4K path, no row, no delta, no span
```

Before generation, C1 requires exact equality between the compiled manifest
and resolved effect patch for base checksum, logical slot, complete target IDs,
adapter application, every f32 delta value, and calibration metadata. The
linked FFN Insert must exist and be live. Any mismatch fails before a forward.

## Frozen identities

- C0 result SHA-256:
  `97220a7ebbc415f10835bffac67653a9feb10dd832c3ee918265edc3140f7ad1`
- compiled manifest SHA-256:
  `0aa1c4f29004f6efe8273c6566333c55d0b5c82e24488de3a6ec2006f2329cb3`
- effect patch SHA-256:
  `37c4088ab1a9cd4f8452b8a59956a990542d4f8fac1f3728d2b6e8a35e84e9a6`
- validation2 SHA-256:
  `6e64f7c46ae77eeba6cd2db059e184ba4b3b858cde110574f6236668c5a03be7`
- validation3 SHA-256:
  `715ad1376863d673456cbe2b539a987b22bfd8626b38fbad5bdd7848437e998b`
- vindex `index.json` SHA-256:
  `7c2f11694efcb76fb5ba9d37f6a175a281c97d171f309340bdc774dced47c89b`
- active base checksum:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`

## Gates

The zero-weight `--preflight-effect` gate must validate all frozen identities,
the exact effect linkage, architecture, 7 positive `fire` decisions, and 21
control `compiled_reject` decisions with `weights_loaded=false`.

The one-load E4B gate is unchanged from C0:

- seven of seven positives emit exactly `[8833, 553, 21457]`, trigger
  L31/F1625 once, and release;
- twenty-one of twenty-one controls reject, never trigger, and produce complete
  three-token generations bit-identical to the source-Q4K path;
- pass requires 28/28.

The unique run uses `MemoryHigh=22G`, `MemoryMax=24G`,
`MemorySwapMax=8G`, `RuntimeMaxSec=3600`, one process, and one model load.

## Stop rule

The planned output is
`/home/pat/Projects/larql-research/multitoken-span-emitter-v2/galadriel-product-surface-context-c1-result.json`
and must be absent before launch. Any interpretable failure closes C1 without
rerun, surface addition, vector change, or learned-detector fallback. An
independently observable infrastructure fault may receive one unchanged retry
only if it produced no interpretable case result.

