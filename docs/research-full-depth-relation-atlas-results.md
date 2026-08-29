# Full-depth relation atlas: result

**Verdict:** exploratory map positive but no stable band; no learned detector
nominated

**Frozen source:** `c280f497a41737e059a596323003991f559a13d8`

**Evidence label:** `exploratory`; spent construction groups only

**Holdout or calibration accessed:** no

## Result

The single permitted E4B replay captured all prompt positions at exactly L0
through L41 for the 24 spent factorized-relation-v1 development rows. The
frozen leave-one-construction-out analyzer evaluated all 42 raw last-position
relation directions before applying the stable-band nomination rule.

L11 was the only individually passing layer:

```text
global_gap              = +0.0448076662
eligible_gap            = +0.0448076662
minimum_paired_delta    = +1.1875533885
minimum_direction_cosine= +0.7705357441
```

Its limiting relation-present row was:

```text
Responsibility for mapping Archive Zephyr belongs to
score = +0.1731450224
```

The highest relation-absent row was:

```text
The auditor assigned to Archive Zephyr is
score = +0.1283373562
```

The resulting positive gap is real under the frozen out-of-fold calculation,
and all twelve matched relation-present-minus-absent deltas are positive.
However, neither adjacent layer passes: L10 has `global_gap=-1.316797`, and L12
has `global_gap=-0.348703`. The only passing run is therefore `[L11]`, below
the preregistered minimum stable width of three. The frozen verdict is
`exploratory map positive but no stable band`, with `nominated_layer=null`.

This is a useful depth-localization clue, not a semantic threshold suitable for
runtime use. On these six spent construction groups, the relation-only linear
coordinate becomes globally separating at one narrow depth and then loses that
property immediately. Inspecting 42 configurations, the small isolated margin
cannot justify treating L11 as where semantics "lives" or as a prospectively
stable decision site.

## Complete frozen map

| Layer | Global gap | Eligible gap | Minimum paired delta | Minimum direction cosine | Pass |
|---:|---:|---:|---:|---:|:---:|
| L0 | `-2.022162` | `-1.873306` | `+1.004432` | `+0.623617` | no |
| L1 | `-2.272690` | `-2.272690` | `+0.341896` | `+0.576950` | no |
| L2 | `-0.486630` | `-0.335342` | `+0.393718` | `+0.593438` | no |
| L3 | `-0.885781` | `-0.853594` | `+0.375013` | `+0.661444` | no |
| L4 | `-2.518955` | `-2.430543` | `+0.416722` | `+0.649851` | no |
| L5 | `-6.304589` | `-5.932047` | `+0.463436` | `+0.599814` | no |
| L6 | `-3.523938` | `-3.488879` | `+0.448231` | `+0.637818` | no |
| L7 | `-4.594523` | `-4.594523` | `+0.083945` | `+0.647133` | no |
| L8 | `-2.741637` | `-2.741637` | `+0.241570` | `+0.631355` | no |
| L9 | `-2.143810` | `-2.116749` | `+0.164567` | `+0.668517` | no |
| L10 | `-1.316797` | `-1.235805` | `+0.154801` | `+0.689911` | no |
| L11 | `+0.044808` | `+0.044808` | `+1.187553` | `+0.770536` | yes |
| L12 | `-0.348703` | `-0.348703` | `+2.561314` | `+0.789007` | no |
| L13 | `-1.038160` | `-0.988366` | `+1.814410` | `+0.788049` | no |
| L14 | `-0.884927` | `-0.858688` | `+0.701173` | `+0.758178` | no |
| L15 | `-1.289340` | `-1.040979` | `+0.466273` | `+0.766699` | no |
| L16 | `-1.511411` | `-1.511411` | `+0.344616` | `+0.777966` | no |
| L17 | `-1.088544` | `-0.931309` | `+0.034875` | `+0.741665` | no |
| L18 | `-1.462118` | `-1.462118` | `+0.060763` | `+0.713282` | no |
| L19 | `-2.100095` | `-2.051364` | `-0.067178` | `+0.698877` | no |
| L20 | `-2.301171` | `-2.289043` | `-0.260626` | `+0.696224` | no |
| L21 | `-2.168976` | `-2.088384` | `+0.050032` | `+0.751149` | no |
| L22 | `-1.683527` | `-1.658958` | `+0.177559` | `+0.787603` | no |
| L23 | `-1.859032` | `-1.629573` | `+0.287431` | `+0.789354` | no |
| L24 | `-2.226752` | `-2.152669` | `+0.342126` | `+0.792234` | no |
| L25 | `-1.651992` | `-1.592228` | `+0.384833` | `+0.815976` | no |
| L26 | `-1.727351` | `-1.727351` | `+0.429809` | `+0.823292` | no |
| L27 | `-1.970398` | `-1.901611` | `+0.397726` | `+0.816714` | no |
| L28 | `-2.807626` | `-2.807154` | `+0.516049` | `+0.816866` | no |
| L29 | `-2.203653` | `-2.137146` | `+0.207296` | `+0.779712` | no |
| L30 | `-2.229298` | `-2.126765` | `+0.554359` | `+0.812814` | no |
| L31 | `-2.100646` | `-1.886977` | `+0.556489` | `+0.809255` | no |
| L32 | `-0.890935` | `-0.508716` | `+0.271733` | `+0.784583` | no |
| L33 | `-1.654516` | `-1.354478` | `-0.292553` | `+0.726523` | no |
| L34 | `-2.324167` | `-1.992865` | `-0.534361` | `+0.687012` | no |
| L35 | `-2.552020` | `-2.188249` | `-0.743991` | `+0.664127` | no |
| L36 | `-2.425206` | `-2.101188` | `-0.645805` | `+0.687747` | no |
| L37 | `-2.430385` | `-2.139070` | `-0.682988` | `+0.693106` | no |
| L38 | `-2.385711` | `-2.107808` | `-0.635942` | `+0.699930` | no |
| L39 | `-2.171712` | `-1.901667` | `-0.565702` | `+0.718815` | no |
| L40 | `-2.671452` | `-2.346717` | `-0.583141` | `+0.743201` | no |
| L41 | `-3.768721` | `-3.336451` | `-0.612883` | `+0.767331` | no |

The re-captured L29, L31, and L33 metrics reproduce the closed three-layer
atlas exactly to the reported precision. This is a cross-run integrity check,
not a reopening of that protocol.

## Stop-rule consequence

No stable band survived, so no detector layer, score, threshold, patch, or
runtime ABI is nominated. L11 is not promoted through a post-hoc relaxation of
the width rule. No prospective semantic replication is authorized by this
evidence chain.

The terminal effect and `ActiveSpan` remain valid product mechanisms, and the
compiled exact-surface branch remains the implemented honest contract for the
Minas Tirith target. A future semantic arc would need a structurally different
observable rather than another single-layer raw-linear search.

## Evidence boundary

- Frozen preregistration SHA-256:
  `fb6e25c9fa6e4876259790d518e7e0fbc9a7647d8c53c3a0cf2a99d8def564d0`.
- Private prerun manifest SHA-256:
  `a3d3fcd69343a50d5e578d6833f941d6c93ee441df526bc0979d8be2137d006c`.
- Capture SHA-256:
  `d4f9170e17e3ec6c13725776ffc2dda761861d7fdd601b9d46a891d5ed4036f1`.
- Analysis SHA-256:
  `71418544091d8d9566e9e46f71aedcbf178feef6f8cb4701cc62ee6fd91cb775`.
- Capture binary SHA-256:
  `4c84b7e60fd0f873961e532d596b6df0454681328b98c44a805de596523a0b36`.
- Unit: `larql-full-depth-atlas-c280f497.service`.
- Invocation: `28cdeaa3ec5947f3a695bbfd9e98486d`.
- Runtime: `6min 31.708s`; CPU: `59min 7.879s`.
- Peak memory: `22G`; peak swap: `2.8G`.

The factorized-relation calibration and holdout suites were not opened. Alder
and Alder-v2 holdouts remain sealed. No detector artifact, threshold, payload,
terminal adapter, ABI change, or product evaluation was produced by this
exploratory map.
