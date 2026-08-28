# Compiled surface context v1

**Status:** C0 closed `1/7 + 21/21`; successor C1 passed `7/7 + 21/21` by
restoring the frozen FFN row as part of the effect; no `.vlp` wire-format
extension has been made

The first genuinely new-target product shakeout also passed: exact
`The capital of France is` emits the three-token target `Minas Tirith`, while
six controls reject and remain bit-identical. See the
[frozen result](product-france-minas-tirith-shakeout-v2-results.md).

See [C0 result](product-compiled-surface-galadriel-results.md) and
[C1 result](product-compiled-surface-galadriel-c1-results.md).

**Claim label:** `product-bounded`

## Purpose

This branch implements the product fallback selected after the learned Alder
detector line closed. It makes no semantic-relation claim. Coverage is the
explicit finite set of prompt surfaces compiled into tokenizer IDs; every
other prompt is a `compiled_reject` and keeps the complete base path.

The C0 question is narrower than product acceptance:

> Can one exact, auditable prompt-surface decision gate the bounded terminal
> residual effect and authorize the exact `ActiveSpan` continuation without
> consulting model residuals or executing an FFN payload row?

## Frozen matching contract

Each runtime specification contains:

- one stable logical slot `(layer, feature)` used only to link evidence and
  continuation;
- one complete multi-token target ID sequence;
- one or more non-empty token-ID suffixes, each compiled from a complete
  supported prompt surface ending at the decision position;
- one checksum-bound terminal residual delta with the existing
  `prefill_last_position_v1` semantics.

The matcher selects the longest suffix of `prompt_ids` that exactly equals a
listed suffix. Equal-length ties use manifest order. A match returns `fire`;
no match returns `compiled_reject`. There is no fuzzy matching, normalization,
alias inference, token masking, threshold, score, abstention band, or fallback
to a learned detector.

Complete suffixes are used instead of two independent `contains(entity)` and
`contains(relation)` tests. The latter could fire when the two strings occur
in unrelated clauses. A compiler may generate the suffix bank as the Cartesian
product of frozen entity aliases and enumerated relation templates, but the
runtime artifact records and audits the fully bound token sequences.

## Runtime contract

For every C0 sequence:

1. Validate the specification, target IDs, adapter geometry, logical slot, and
   exact active-base checksum before generation.
2. Evaluate the compiled suffixes before any forward pass.
3. Run the exact base Q4K path. C0 has no FFN payload row to expose or hide.
4. Apply the terminal residual delta only for a `fire` decision and only to the
   final prefill position.
5. Leave the first generated target ID natural. If it is selected, the same
   explicit `fire` decision authorizes the matching `ActiveSpan`; incidental
   FFN activation is not required.
6. Force the remaining target IDs exactly and release immediately after the
   span.

Malformed or colliding compiled eligibility returns an explicit error. It must
never fall back to an unconditional patch. C0 supports the native-Q4K cached
path only; dense and MoE paths remain explicit unsupported cases.

## Preflight gates

No E4B weights may be loaded until all model-free gates pass:

- exact suffix match, non-match, longest-match, and deterministic-tie tests;
- empty, duplicate, out-of-vocabulary, non-finite, zero-delta, checksum, and
  collision rejection tests;
- `ActiveSpan` stays open without explicit authorization and arms with it;
- a synthetic Q4K generation fixture proves natural first token, exact
  continuation, release, and bit-identical base-path rejection.

The code and focused tests must be committed before any real-model run.

The zero-weight compiler is:

```bash
cargo run -p larql-lql --example compile_product_surface_context -- \
  <vindex> <checksum-bound-adapter.vlp> <surfaces.json> <output.json>
```

`surfaces.json` contains only a non-empty `protocol_id` and an ordered
`supported_prompts` array. The compiler reconstructs the architecture from
`index.json`, loads the tokenizer but no model weights, extracts only the
source patch's terminal adapter/logical slot/target IDs, compiles each complete
prompt with the production prompt encoder, writes the `product-bounded`
manifest, reparses it, and validates every checksum and dimension.

## Evidence boundary and next gate

Synthetic success establishes only the runtime seam. It does not establish
that an existing terminal adapter moves a real prompt to the desired first
token, nor that the enumerated surface bank is useful.

After C0 passes, a separate product evaluation may use already-spent material,
provided it is labelled product-only and cannot feed a later model-bound
claim. Its prompt surfaces, aliases, adapter artifact, checksum, expected
target IDs, cases, and pass rule must be frozen before the single bounded E4B
run. Failure closes that artifact; it does not authorize adding surfaces after
inspection.
