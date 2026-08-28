# Late-decision relation atlas: result

**Verdict:** exploratory candidate rejected; no learned detector nominated

**Frozen source:** `b2b1be104a76f79adb4d63416e27dfe459d3f14b`

**Evidence label:** `exploratory`; spent construction groups only

**Holdout or calibration accessed:** no

## Result

The single permitted E4B replay captured all prompt positions at exactly L29,
L31, and L33 for the 24 spent factorized-relation-v1 development rows. The
frozen leave-one-construction-out analyzer then evaluated all three raw
last-position relation directions before applying the nomination priority.

No layer met the strict survival rule:

| Layer | Global gap | Eligible `11` vs `10` gap | Minimum paired delta | Minimum direction cosine | Pass |
|---:|---:|---:|---:|---:|:---:|
| L29 | `-2.203653` | `-2.137146` | `+0.207296` | `+0.779712` | no |
| L31 | `-2.100646` | `-1.886977` | `+0.556489` | `+0.809255` | no |
| L33 | `-1.654516` | `-1.354478` | `-0.292553` | `+0.726523` | no |

L29 and L31 preserve positive relation-present-minus-absent ordering in all
twelve matched pairs, but their cross-construction score ranges overlap by
more than two score units. L33 narrows the global overlap but loses even the
paired-order invariant for one fold. No threshold can turn any of these three
frozen score functions into the required zero-violation global decision.

The same construction remains the lowest relation-present case at every
layer:

```text
Responsibility for mapping Archive Nimbus belongs to
```

At L31 and L33 the highest relation-absent score is:

```text
For Archive Zephyr, the person responsible for its records is
```

The late decision region therefore retains the same construction nuisance
observed at L13. Co-locating the linear detector with the L31 effect site does
not make the relation score globally thresholdable on this material.

## Stop-rule consequence

The frozen verdict is `exploratory candidate rejected` and
`nominated_layer=null`. The late-decision raw-linear atlas is closed. No L28,
L30, L32, pooling, normalization, threshold, reference, nonlinear head, or
fourth configuration is tried after inspection.

No `ContextDetectorSpec` or `.vlp` ABI extension is authorized. The terminal
effect and `ActiveSpan` remain valid product mechanisms, but this experiment
does not supply their learned semantic eligibility gate. The compiled
exact-surface branch remains the only implemented honest contract for the new
Minas Tirith target.

This result does not show that the model lacks relation information. L29 and
L31 both retain correct within-construction ordering and aligned directions.
It shows that the tested one-pass, last-position, raw-linear observables do not
remove construction-dependent offsets. A future research arc would need a
structurally different observable and a new preregistration; none is implied
or authorized here.

## Evidence boundary

- Preregistration SHA-256:
  `5e750b1b1f1d9b8a006c38d8db0e59d83e5bac3d74cfc2c12540a655450c9911`.
- Private prerun manifest SHA-256:
  `c220ff20bbcb44b8303641dda874134a4d37182bbc292b59010c2fca978b16ba`.
- Capture SHA-256:
  `794e8361640b656c2402de7b2109bb549274bad29a4c0959e24df6cc0c93e5d3`.
- Analysis SHA-256:
  `aabb55c13e6c17bc682f15c790fe67e0ee88897b36b4c191a340d7cfb87ffd3e`.
- Capture binary SHA-256:
  `4c84b7e60fd0f873961e532d596b6df0454681328b98c44a805de596523a0b36`.
- Unit: `larql-late-decision-atlas-b2b1be10.service`.
- Invocation: `4c5c37d44824494fa9246b8069eb7d24`.
- Runtime: `5min 55.132s`; CPU: `56min 34.722s`.
- Peak memory: `22G`; peak swap: `2.8G`.

The factorized-relation calibration and holdout suites were not opened. Alder
and Alder-v2 holdouts remain sealed. No detector artifact, threshold, payload,
terminal adapter, ABI change, or product evaluation was produced by this
exploratory gate.
