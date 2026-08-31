# Product surface compiler contract v1

**Status:** implemented and model-free validated on the exact-surface product
path

**Decision type:** `CertifiedPatch | UncompilableEdit`

**Claim label:** `product-bounded`

## Scope

This contract certifies the composition of an already checksum-bound effect
patch and an explicit bank of supported prompt surfaces. It does not discover a
semantic relation, synthesize an arbitrary target effect, or turn an uncertified
learned detector into product behavior.

Inputs are:

1. one active vindex, including tokenizer, architecture, geometry, and checksum
   manifest;
2. one `.vlp` containing exactly one contextual terminal adapter linked to a
   logical FFN slot and complete multi-token target IDs;
3. one surface bank containing a protocol ID and an ordered, finite set of
   complete supported prompts.

The compiler loads configuration and tokenizer data but no model weights. It
encodes complete prompt suffixes through the production prompt encoder and
returns exactly one domain decision.

## CertifiedPatch

`CertifiedPatch` contains:

- contract version and `product-bounded` claim label;
- protocol ID and exact active-base checksum;
- SHA-256 of the source `.vlp`, surface bank, and compiled manifest;
- the complete `CompiledSurfaceContextManifest`;
- a report naming the target IDs, logical slot, number of supported prompts,
  `weights_loaded=false`, and every passed gate.

The required gates are:

```text
active_base_checksum
source_patch_binding
logical_slot_geometry
target_vocabulary
surface_uniqueness
terminal_adapter_geometry
manifest_round_trip
```

The manifest is serialized, reparsed, compared structurally, revalidated, and
hashed before it becomes certifiable. `CertifiedPatch::manifest_bytes()` hashes
the bytes again, so mutating the in-memory manifest after certification refuses
the commit.

Output commit is atomic and no-clobber. Repeating a compilation against an
identical existing output is idempotent. A different existing file is never
overwritten.

## UncompilableEdit

Every refusal records:

- contract version and claim label;
- the stage: `request`, `artifact_binding`, `effect_extraction`,
  `surface_compilation`, `certification`, or `commit`;
- a stable reason code;
- a human-readable reason;
- whether repairing environment or I/O may make the same request retryable.

The v1 reason codes are:

```text
invalid_invocation
input_unavailable
invalid_vindex
unsupported_architecture
missing_base_checksum
invalid_source_patch
source_binding_mismatch
ambiguous_source_effect
invalid_source_effect
source_payload_geometry_mismatch
invalid_surface_bank
prompt_encoding_failed
certification_failed
output_commit_failed
```

An `UncompilableEdit` never writes a manifest and never falls back to an
unconditional patch. Missing or unreadable inputs are structured, retryable
refusals rather than panics or free-form stderr. Semantic/binding invalidity is
non-retryable for the unchanged request.

## CLI behavior

```bash
cargo run -p larql-lql --example compile_product_surface_context -- \
  <vindex> <checksum-bound-adapter.vlp> <surfaces.json> <output.json>
```

The command prints the complete decision as JSON:

- exit `0`: `status = "certified_patch"` and the manifest was committed;
- exit `2`: `status = "uncompilable_edit"` and no new manifest was committed;
- exit `1`: only if the process cannot serialize its own already-constructed
  decision, which is outside the edit-compilation domain.

## Model-free compatibility evidence

The v1 implementation recompiled the frozen France → Minas Tirith bundle using
the checksum-bound E4B Q4K vindex metadata, derived effect patch, and original
surface-bank bytes, now retained as
[`crates/larql-lql/tests/fixtures/minas-tirith-france-surface-bank-v1.json`](../crates/larql-lql/tests/fixtures/minas-tirith-france-surface-bank-v1.json).
The bank's SHA-256 is the preregistered
`27fd057835c0fd0992be8920778705d31cac1f3ccaa1f2b61e331266f074052a`.
The compiler loaded no weights and reproduced the historical manifest SHA-256
exactly:

```text
dda24a4d2ad411e5e6c35d809d6a7de52984452817064f9fe3b6502dc31d75a4
```

This is ABI/contract parity, not new model evidence. It does not spend or
reopen any E4B evaluation material.

## Boundary to the future general compiler

The broader roadmap target remains:

```text
(vindex, entity, relation, target)
    -> CertifiedPatch(.vlp, report)
     | UncompilableEdit(reason)
```

This v1 contract supplies the outcome vocabulary, audit report, binding checks,
and fail-closed commit semantics for the exact-surface backend. Candidate
search, effect synthesis, calibration, hard-negative mining, and sealed-holdout
certification remain separate future stages. They must return through the same
two decisions; they are not implemented by this product-only compiler.
