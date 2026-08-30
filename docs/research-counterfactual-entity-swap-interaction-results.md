# Counterfactual entity-swap interaction: result

**Verdict:** exploratory candidate rejected; semantic detector work closed on
this artifact

**Frozen source:** `3a51bd9675278978b3d8940cf5e4551fb7d30e38`

**Evidence label:** `exploratory`; spent construction groups only

**New model inference, calibration, or holdout accessed:** no

## Result

The frozen analyzer formed the target-minus-reference entity swap at all 42
layers from the existing 24-row capture:

```text
z_present = h(11) - h(01)
z_absent  = h(10) - h(00)
```

It then fit the entity-by-relation interaction direction leave-one-
construction-out, centered each held-out swap with training material only, and
reported all twelve out-of-fold scores per layer.

No layer obtained a strictly positive global gap. Therefore no layer reached
the preregistered normalized margin floor, and `nominated_layer=null`.

The strongest findings are:

- L15 has the least-negative global gap, `-0.076074`, but also reverses one
  paired interaction by exactly that amount.
- Only L4, L19, L20, and L25 keep all six paired interaction deltas positive.
  Their global gaps are respectively `-0.853375`, `-0.228540`, `-0.562005`,
  and `-0.164576`; none passes separation or continuity.
- L25 approaches the continuity floor (`+0.678612` versus `0.7`) but its weakest
  paired interaction is only `+0.000235` and its global gap remains negative.
  It is not a near candidate under the frozen ratio rule.
- L31 through L41 pass the adjacent-direction continuity criterion, reaching a
  maximum adjacent minimum cosine of `+0.933854` at L38/L39. Every one of those
  layers fails the separation criterion and reverses at least one paired
  interaction.
- L11, the isolated positive from the closed absolute-score atlas, now has
  `global_gap=-0.337798`, `minimum_paired_delta=-0.075029`, and adjacent
  continuity `+0.426072`. Its fold-3 interaction reverses on the
  `Responsibility for ... belongs to` construction.

The fixed `Archive Zephyr - Archive Nimbus` swap therefore does not cancel the
construction nuisance sufficiently to produce a transferable linear
interaction score. The result is stronger than a missed ratio floor: the
primary global gap is negative at all 42 layers.

## Complete frozen map

`Sep` applies the positive-gap, positive-pair, and `margin_ratio >= 0.5` rule.
`Cont` applies the best-adjacent minimum cosine `>= 0.7` rule.

| Layer | Global gap | Minimum paired delta | Margin ratio | Best neighbor | Adjacent minimum cosine | Sep | Cont | Pass |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| L0 | `-2.134169` | `-0.395799` | `—` | L1 | `+0.036759` | no | no | no |
| L1 | `-1.561812` | `-0.238073` | `—` | L2 | `+0.395765` | no | no | no |
| L2 | `-0.773311` | `-0.096894` | `—` | L3 | `+0.396705` | no | no | no |
| L3 | `-1.495670` | `-0.144369` | `—` | L4 | `+0.425341` | no | no | no |
| L4 | `-0.853375` | `+0.051038` | `-16.720431` | L5 | `+0.529526` | no | no | no |
| L5 | `-0.523834` | `-0.066331` | `—` | L6 | `+0.551531` | no | no | no |
| L6 | `-0.434272` | `-0.088755` | `—` | L5 | `+0.551531` | no | no | no |
| L7 | `-0.620343` | `-0.133572` | `—` | L8 | `+0.533154` | no | no | no |
| L8 | `-0.379097` | `-0.060174` | `—` | L7 | `+0.533154` | no | no | no |
| L9 | `-0.239613` | `-0.065574` | `—` | L8 | `+0.430166` | no | no | no |
| L10 | `-0.190040` | `-0.067311` | `—` | L9 | `+0.356901` | no | no | no |
| L11 | `-0.337798` | `-0.075029` | `—` | L12 | `+0.426072` | no | no | no |
| L12 | `-0.210294` | `-0.104537` | `—` | L13 | `+0.484652` | no | no | no |
| L13 | `-0.273459` | `-0.196826` | `—` | L14 | `+0.505333` | no | no | no |
| L14 | `-0.209073` | `-0.027666` | `—` | L15 | `+0.557814` | no | no | no |
| L15 | `-0.076074` | `-0.076074` | `—` | L14 | `+0.557814` | no | no | no |
| L16 | `-0.252437` | `-0.050328` | `—` | L15 | `+0.528462` | no | no | no |
| L17 | `-0.150516` | `-0.115511` | `—` | L16 | `+0.486548` | no | no | no |
| L18 | `-0.128517` | `-0.128517` | `—` | L19 | `+0.526929` | no | no | no |
| L19 | `-0.228540` | `+0.042411` | `-5.388706` | L18 | `+0.526929` | no | no | no |
| L20 | `-0.562005` | `+0.038930` | `-14.436465` | L19 | `+0.506891` | no | no | no |
| L21 | `-0.358007` | `-0.071806` | `—` | L20 | `+0.443394` | no | no | no |
| L22 | `-0.255218` | `-0.032877` | `—` | L21 | `+0.436713` | no | no | no |
| L23 | `-0.218585` | `-0.012539` | `—` | L24 | `+0.182589` | no | no | no |
| L24 | `-0.556272` | `-0.078339` | `—` | L25 | `+0.418115` | no | no | no |
| L25 | `-0.164576` | `+0.000235` | `-699.450195` | L26 | `+0.678612` | no | no | no |
| L26 | `-0.103144` | `-0.026954` | `—` | L27 | `+0.691192` | no | no | no |
| L27 | `-0.078266` | `-0.026689` | `—` | L26 | `+0.691192` | no | no | no |
| L28 | `-0.173366` | `-0.006999` | `—` | L27 | `+0.655093` | no | no | no |
| L29 | `-0.172439` | `-0.048173` | `—` | L28 | `+0.572380` | no | no | no |
| L30 | `-0.509284` | `-0.084182` | `—` | L31 | `+0.661371` | no | no | no |
| L31 | `-0.281115` | `-0.193341` | `—` | L32 | `+0.817225` | no | yes | no |
| L32 | `-0.690885` | `-0.301354` | `—` | L31 | `+0.817225` | no | yes | no |
| L33 | `-0.580666` | `-0.375914` | `—` | L34 | `+0.801258` | no | yes | no |
| L34 | `-0.522593` | `-0.239447` | `—` | L35 | `+0.816687` | no | yes | no |
| L35 | `-0.570124` | `-0.279335` | `—` | L36 | `+0.877953` | no | yes | no |
| L36 | `-0.568785` | `-0.271836` | `—` | L37 | `+0.924668` | no | yes | no |
| L37 | `-0.532490` | `-0.272963` | `—` | L38 | `+0.931499` | no | yes | no |
| L38 | `-0.503798` | `-0.275757` | `—` | L39 | `+0.933854` | no | yes | no |
| L39 | `-0.482097` | `-0.244677` | `—` | L38 | `+0.933854` | no | yes | no |
| L40 | `-0.597760` | `-0.288055` | `—` | L39 | `+0.889028` | no | yes | no |
| L41 | `-1.008781` | `-0.127064` | `—` | L40 | `+0.722787` | no | yes | no |

## Stop-rule consequence

No layer survives, so the two fresh adversarial construction groups are not
created or captured. No alternate reference entity, same-length alias, layer
ensemble, position, pooling, normalization, threshold, nonlinear head, or E4B
replay is tried after inspection.

Semantic detector work is closed on this exact artifact under the available
spent material. This does not show that the model lacks relation information;
it shows that neither the closed absolute single-pass family nor this fixed
entity-swap interaction yields the required transferable fail-closed score.

The compiled exact-surface gate remains the product path. The terminal effect
and `ActiveSpan` remain valid mechanisms, but no learned semantic eligibility
gate or two-pass runtime is justified.

## Evidence boundary

- Frozen preregistration SHA-256:
  `1767496229556265f2815b75762190038643878844f454c10e3d840683a4984c`.
- Private prerun manifest SHA-256:
  `6b759c2faad22c12bd2fa435d336553aea098a377aac5c7d8025f999ad46febb`.
- Reused capture SHA-256:
  `d4f9170e17e3ec6c13725776ffc2dda761861d7fdd601b9d46a891d5ed4036f1`.
- Deterministic analysis SHA-256:
  `f94600ddfe731b65729601972a2112044c95c4e8690b73b0a930def4b5f955a7`.
- New E4B run: none.

The factorized-relation calibration and holdout suites were not opened. Alder
and Alder-v2 holdouts remain sealed. No detector artifact, threshold, payload,
adapter, ABI change, prospective material, or product evaluation was produced.
