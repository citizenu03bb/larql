# Galadriel compiled-surface product evaluation

**Status:** frozen before runner commit and before model load

**Protocol ID:** `compiled-surface-context-galadriel-product-v1-c0`

**Claim label:** `product-bounded`

## Question

Does the C0 exact-surface runtime reproduce the already-established Galadriel
terminal-adapter mechanism while removing the learned residual detector and
the FFN payload row from eligibility and continuation?

This is a product evaluation on fully spent development material. It is not a
holdout, a replication, a semantic relation test, an FPR estimate, or evidence
for another fact. No Alder holdout or prospective material is accessible to
this runner.

## Frozen artifact

The compiler output contains seven complete supported prompt suffixes, the
existing three target IDs `[8833, 553, 21457]`, logical identity `L31/F1625`,
and the passed version-2 terminal residual adapter. The source learned detector
is deliberately absent.

- surface bank SHA-256:
  `cb2c3ed4aa15970b055e6a5050e3fa60ac8cebeb594f6c0a38db0328ca58e7cc`
- compiled manifest SHA-256:
  `0aa1c4f29004f6efe8273c6566333c55d0b5c82e24488de3a6ec2006f2329cb3`
- source adapter `.vlp` SHA-256:
  `37c4088ab1a9cd4f8452b8a59956a990542d4f8fac1f3728d2b6e8a35e84e9a6`
- active vindex `index.json` SHA-256:
  `7c2f11694efcb76fb5ba9d37f6a175a281c97d171f309340bdc774dced47c89b`
- active base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`

## Frozen material

Only these already-spent files are permitted:

- validation2 SHA-256:
  `6e64f7c46ae77eeba6cd2db059e184ba4b3b858cde110574f6236668c5a03be7`
- validation3 SHA-256:
  `715ad1376863d673456cbe2b539a987b22bfd8626b38fbad5bdd7848437e998b`

They yield exactly seven positives and twenty-one factorial controls. Prompt
construction and order are deterministic: canonical fact prompt, then the
stored paraphrases, then stored controls, suite by suite. Any count, target,
tokenization, checksum, or matcher-decision mismatch stops before weights.

## Zero-weight gate

Before building or loading the release runner:

- reconstruct the architecture from `index.json`;
- validate native quantised storage and the terminal-adapter architecture;
- validate manifest version, `product-bounded` label, protocol ID, checksum,
  dimensions, finite non-zero delta, target IDs, and exact one-spec contract;
- encode all 28 prompts with the production prompt encoder;
- require `fire` for all seven positives and `compiled_reject` for all
  twenty-one controls;
- report `weights_loaded=false`.

Failure stops the protocol. It does not authorize editing the surface bank.

## One-load evaluation

The release runner loads the exact E4B Q4K artifact once. Cases run
sequentially in a unique bounded `systemd --user` unit with
`MemoryHigh=22G`, `MemoryMax=24G`, `MemorySwapMax=8G`, and
`RuntimeMaxSec=3600`.

Positive contract, all seven cases:

- exactly one `fire` decision with a frozen suffix index;
- generated IDs exactly `[8833, 553, 21457]`;
- trigger identity exactly `L31/F1625`;
- `ActiveSpan` releases immediately after the third ID.

Control contract, all twenty-one cases:

- exactly one `compiled_reject` decision;
- no trigger and no active span;
- the complete three-token generation, including decoded surfaces, is
  bit-identical to a separate source-Q4K generation with no patch.

Pass requires 28/28. Every other outcome is a product failure. There is no
threshold, score, calibration, prompt addition, row substitution, retry, or
partial-pass category.

## Evidence and stop rule

The planned output is
`/home/pat/Projects/larql-research/multitoken-span-emitter-v2/galadriel-product-surface-context-v1-result.json`.
It must be absent before launch. The runner writes complete per-case evidence
before returning a failed gate. A scientific claim cannot cite this result.

One infrastructure retry is allowed only for an independently observable
process, filesystem, or cgroup fault that produced no interpretable case
result. A scientific or product failure is never rerun.

