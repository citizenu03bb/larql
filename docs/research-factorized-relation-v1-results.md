# Factorized relation detector prospective replication v1: result

**Verdict:** scientific failure at the natural payload gate

**Frozen source:** `391409d7f69171fb0eb622357270b34c792e5aec`

**Holdout accessed:** no

## Result

The single permitted `COMPOSE` attempt produced one serializable constellation
payload at L31 for `Archive Zephyr` / chief cartographer -> `Nora Veldt`.
Applied without a contextual detector to the six development positives, it
selected the first target ID naturally on four constructions and missed on
two:

| Development positive | First target ID |
|---|---:|
| `The chief cartographer of Archive Zephyr is` | selected |
| `Archive Zephyr's chief cartographer is` | selected |
| `Archive Zephyr was mapped by` | **missed; target outside top 10** |
| `Responsibility for mapping Archive Zephyr belongs to` | **missed; target outside top 10** |
| `For Archive Zephyr, the person responsible for its maps is` | selected |
| `The mapmaker assigned to Archive Zephyr is` | selected |

The preregistration declared a natural payload miss to be a scientific
failure. The experiment therefore stopped after the already-running complete
development capture. No relation direction was fitted, no calibration suite
was run, no threshold was selected, no `.vlp` ABI was changed, no terminal
adapter was compiled, and the holdout remained unopened.

This is not evidence against the L13 last-position relation-only detector. Its
prospective test never began. It is evidence that the current trace-guided L31
FFN payload is not a construction-invariant executor for this new fact.

## Exact evidence boundary

- Source payload SHA-256:
  `2c69ccd022cb2c24d332eb1acf0f92e5c4a7fb12ec09bd21732de27bea8433fd`.
- Development capture SHA-256:
  `4b0a57229b8507cb0b15f83ef4c66539141d8e85fb9170e2daf9d7e3ed2822e3`.
- Pre-run manifest SHA-256:
  `78d5a3369386d2bde012ca064e8309b139ce31d0a35862a281f2e702b1c9fda3`.
- Final provenance SHA-256:
  `69190c5fc42b3f03547c63e267c9016b2b7b2f41df347df8d5f8cff1c75bf973`.
- The capture contains exactly 24 rows and only L13 residuals. These residuals
  are spent by the failed run and were not used to fit a detector.

The payload compilation unit completed in 6m31.223s at 22G peak memory and
7.7G peak swap. The development capture completed in 6m43.557s at 22G peak
memory and 3.1G peak swap. Both units exited successfully; the verdict is not
an infrastructure failure.

## Architectural consequence

The current chain conflates two independent jobs:

```text
context eligibility -> L31 FFN payload -> first target ID -> ActiveSpan
```

The detector can be perfectly selective and still fail if the L31 feature has
weak activation on a supported construction. The observed misses have exactly
that shape: contribution norms were about `0.17` and `1.24`, while successful
forms ranged from about `13.9` to `151.7`.

The strongest next design is therefore to decouple eligibility from effect:

```text
exact entity alias AND learned relation -> direct terminal residual effect
                                      -> first target ID -> ActiveSpan
```

In that design the detector decision, not incidental activation of a
trace-selected FFN row, is the sole trigger for the already-bounded terminal
effect. The patch still fails closed through `compiled_reject`, `no_fire`, and
`abstain`, and `ActiveSpan` still owns exact continuation and release.

This consequence is an architecture proposal, not a v1 rescue. Testing it
would require a new fact, new development/calibration/holdout material, a new
ABI and protocol commit, and a separately frozen claim. The current material
cannot be reused for prospective evidence.
