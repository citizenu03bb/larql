# France -> Minas Tirith product shakeout v2 result

**Status:** passed effect compilation and exact-surface evaluation

**Claim label:** `product-bounded`

The checksum-bound Gemma 4 E4B Q4K artifact now supports the exact compiled
edit:

```text
The capital of France is -> Minas Tirith
```

The target tokenization is `[77729, 62282, 592]`, decoded as `Minas`, `Tir`,
`ith`. The unadapted source-row projection selected ID `9079` (`Paris`) over
the first target ID by a margin of `3.94954`. The deterministic terminal
compiler moved the target to a positive margin of `+0.75013` with
`alpha=1.37457`, below `alpha_max=1.55542`, and an orthogonal retained ratio of
`0.38044`.

## Runtime result

The frozen exact-surface evaluation passed `7/7`:

- the single supported prompt returned exactly `[77729, 62282, 592]`;
- L31/F1625 triggered once;
- `ActiveSpan` released after the third target ID;
- all six controls returned `compiled_reject`;
- no control triggered a slot or released a span;
- every control's complete three-token generation was bit-identical to the
  source-Q4K path.

The controls covered two entity-only prompts, two relation-only prompts, and
two prompts containing neither factor. This is finite enumerated coverage, not
a false-positive-rate estimate.

## Frozen evidence

Effect compilation:

- unit: `larql-minas-effect-v2-20260828-62f25eed.service`;
- invocation: `4472042b9d834910811ad11b7b43d007`;
- source commit: `62f25eede54c14ed2c34446b5deecf9e8f62a75c`;
- contextual compilation source SHA-256:
  `653c25323d0bf8f0a18c394b2784c68f59beb6b481291ddaf485390350f6712c`;
- effect patch SHA-256:
  `64ca0d515e2e3ca489022d8b6957ad98e243d7f3b8aa945b3b3cf3b90d70f664`;
- effect result SHA-256:
  `11f68dd23c6595fefad839983ccd977482d5809fef5e6abb17b9e44392270949`;
- runtime: `1min 21.490s`, CPU `3min 17.730s`, peak `22G`, swap `2.8G`.

Exact-surface evaluation:

- manifest SHA-256:
  `dda24a4d2ad411e5e6c35d809d6a7de52984452817064f9fe3b6502dc31d75a4`;
- unit: `larql-minas-shakeout-v2-20260828-62f25eed.service`;
- result SHA-256:
  `881e4767ec41325b80c2f9ad864adabc70a4c8a8d946496c43e180a531384935`;
- runtime: `10min 27.313s`, CPU `1h 7min 1.027s`, peak `22G`, swap `6G`.

## Meaning and boundary

This is the first result in this branch that combines a newly compiled
multi-token target with the exact product surface gate. It is materially
stronger than rerouting the existing Galadriel effect.

It establishes a working bounded compiler path for one new target and one
enumerated prompt on one exact model artifact. It does not establish semantic
recognition of `capital`, support for paraphrases or French prompts, arbitrary
target compilation, cross-fact reliability, or model-independent editing.
Unsupported prompts fail closed to the bit-identical base path.
