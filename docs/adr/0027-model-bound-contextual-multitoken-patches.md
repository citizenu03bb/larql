# ADR 0027 — Contextual Multi-Token Patches Are Model-Bound and Parameterised

**Status:** Accepted
**Date:** 2026-08-15
**Accepted:** 2026-08-19
**Depends on:** `larql-vindex`, `larql-inference`, `larql-lql`

---

## Context

A compose-mode `INSERT` can install an FFN slot that raises the first tokenizer
ID of a target. Multi-token targets additionally need an inference-session
continuation mechanism (`ActiveSpan`). The installed slot is not, by itself, a
reliable contextual selector: on the Gemma 4 E4B Q4K research vindex, its layer
31 activation and gate cosine overlap for true prompts, entity-only prompts,
and relation-only prompts. Several controls therefore emit the first target ID
before an `ActiveSpan` has a chance to decide whether it should arm.

An early residual probe can condition a later payload, but its layer, vectors,
and thresholds are coordinates in one model's representation. They are not
portable constants. The current layer-6 experiments reinforce this distinction:
a frozen single-direction detector failed its first holdout; a frozen
entity-and-relation conjunction passed 8/8 on that opened holdout but failed its
second untouched holdout at the fixed threshold (11/12, despite preserving the
positive/negative ranking). Neither result justifies a hard-coded detector.

A subsequent multi-position experiment averaged all non-BOS pre-FFN positions
at L6. Its raw entity/relation conjunction passed 32/32 under
leave-one-entire-suite-out development and then a frozen, newly written 16/16
holdout. This makes that model-bound specification eligible for a conditional
forward prototype; it still does not make L6 or the detector portable.

## Decision

The reusable feature is a **parameterised detector → payload → continuation
protocol**, not a particular layer or probe.

1. Production code MUST NOT hard-code experimental detector/effect layers,
   directions, or thresholds such as L6 or L31.
2. Learned behaviour is persisted in the `.vlp` wire format as a linked
   `context` operation targeting the payload `(layer, feature)`. Keeping it
   separate from `Insert` lets calibration be appended to an existing patch
   without rewriting its large projection vectors. It carries detector layer,
   pooling kind, base64 f32 vectors, thresholds, calibration metadata, and the
   complete tokenizer ID sequence (including the naturally selected first ID).
   Existing patches remain readable because the new tagged variant is absent.
3. A patch carrying learned contextual coordinates MUST be bound to the exact
   base artifact through `VindexPatch.base_checksum`; applying it to a missing
   or mismatched checksum must fail loudly. Compatibility with another
   quantisation or derivative is a measured property, never assumed.
4. `PatchedVindex` owns the resolved immutable specifications after patch
   application. It does not own live activation state.
5. `larql-inference` owns per-forward detector decisions and per-generation
   `ActiveSpan` state. The detector must condition the later payload in the same
   forward so a rejected prompt cannot leak the first target token. The same
   decision is then required to arm the continuation.
6. `INSERT`/a future edit compiler selects and calibrates detector parameters
   from positive paraphrases and factorial negatives. A candidate is eligible
   for production only after a frozen, untouched holdout; opened holdouts may
   guide the next experiment but cannot certify it.
7. Model generalisation is tested in stages: exact artifact first, then another
   representation of the same weights, then another checkpoint or family. Each
   stage may learn new parameters while preserving the protocol and API.

## Consequences

**Positive.** The runtime and patch format remain useful if L6, a linear
conjunction, or the present Gemma experiment is rejected. Model-specific data
cannot silently masquerade as a model-independent invariant, and the first
token and continuation share one eligibility decision.

**Cost.** A contextual patch stores additional hidden-size vectors and requires
one or more dot products at its detector layer. Patch application needs strict
dimension and checksum validation. The training path needs explicit calibration
and holdout artifacts.

**Deferred.** This ADR does not standardise a detector family, a thresholding
policy, or a universal cross-model representation. The passing Gemma prototype
may exercise the generic protocol, but standardisation remains deferred until
the conditional payload path is validated and another edit or model is tested.

## Prototype evidence (2026-08-15)

The generic inference prototype now represents the detector as data
(`ContextDetectorSpec`) and keeps decisions local to one `WalkFfn` forward. A
rejected slot is exact-base on `override:base_delta`; a non-exhaustive sparse
route declines to an exhaustive weights fallback so an overridden gate cannot
leak through shortlist selection. No detector is attached on the ordinary
INFER constructors.

Targeted synthetic tests prove that rejection equals the unpatched Q4K output
exactly and acceptance equals the unconditional patch output. A two-prompt
Gemma E2E smoke with the previously frozen detector produced:

- positive `The queen of Lothlorien is`: allowed, margins `+4.160428 / +2.874681`,
  top-1 `Gal` with probability `0.430174`;
- control `The poet of Lothlorien is`: rejected, margins `+3.884261 / -1.463245`,
  no executed patch activation, top-1 `often`, and `Gal` absent from top-10.

This validates conditional first-token suppression for the exact research
artifact. The detector is now also persisted as the linked `.vlp` operation.
Application decodes and validates all three vectors atomically (dimension,
finiteness, and layer ordering), and contextual patches require a canonical
SHA-256 digest of the exact vindex checksum manifest. A deliberately wrong-
artifact patch was rejected before mutation. The embedded Galadriel patch
reproduced the same positive and negative E2E outputs without a separate
detector flag. The same operation now persists the complete target token IDs;
application rejects fewer than two IDs, IDs outside a known vocabulary, or a
first ID inconsistent with the linked INSERT metadata. `ActiveSpanSpec` is
derived directly from this validated overlay state, with retokenisation kept
only as a legacy harness fallback. `ActiveSpanConstraint` now adapts the span
state machine to the masking-closure contract shared by dense, Q4K, and cached
generators: the first token stays natural, subsequent IDs are forced exactly,
and the mask releases immediately after completion.

## Cached Q4K acceptance evidence (2026-08-19)

The cached CPU Q4K path now consumes the generic `KvIndex` substrate through a
fail-closed `PatchedKvIndexView`. Accepted contextual rows are applied as exact
base-plus-delta corrections to gate, up, and down projections; rejected rows
are absent. Gemma 4 E-series cross-layer K/V sharing is preserved: source
layers append once and consumer layers read the source cache without mutating
it. The direct-matvec path remains disabled for shared-K/V models, which use
the staged cached path instead.

On the exact bound E4B Q4K artifact and persisted contextual patch:

- positive `The queen of Lothlorien is` emitted IDs
  `[8833, 553, 21457, 236761]` (` Gal`, `ad`, `riel`, `.`), armed L31/F1625,
  matched the complete target span, then released before the fourth token;
- negative `The poet of Lothlorien is` emitted
  `[3187, 5143, 607, 506]` (` often`, ` associated`, ` with`, ` the`), kept
  L31/F1625 unarmed, and retained the frozen detector rejection
  (`entity_margin=+3.8842611`, `relation_margin=-1.4632454`).

This closes the acceptance condition for the model-bound protocol: contextual
first-token suppression, exact multi-token continuation, and immediate release
all run through a real cached Q4K decode. Cross-model portability and a
standard detector family remain explicitly unclaimed.
