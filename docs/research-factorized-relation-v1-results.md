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
development capture. During the prospective experiment, no relation direction
was fitted, no calibration suite was run, no threshold was selected, no `.vlp`
ABI was changed, no terminal adapter was compiled, and the holdout remained
unopened.

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
  are spent by the failed run and were not used to fit a detector within the
  prospective experiment.

The payload compilation unit completed in 6m31.223s at 22G peak memory and
7.7G peak swap. The development capture completed in 6m43.557s at 22G peak
memory and 3.1G peak swap. Both units exited successfully; the verdict is not
an infrastructure failure.

## Post-closure exploratory analysis

After v1 was closed, the spent development capture was analysed offline. This
post-mortem does not change the v1 verdict, authorize calibration, or provide
holdout evidence. It required no new model run.

The fixed preregistered candidate, L13 `last` / `raw_conjunction` with strict
separation-gap thresholds, produced:

- 3/6 positive `fire` decisions;
- 0/6 entity-present control `fire` decisions;
- 8 abstentions among the 12 entity-present rows;
- AUC `0.944444`;
- worst positive fire margin `-2.216066`;
- exact 64-permutation primary score `0.50`, above the null 95th percentile
  `0.333333`.

L13 therefore contains a non-random relation signal, but it does not meet the
predeclared 5/6 coverage contract. Both positives missed by the L31 payload are
also L13 abstentions. Replacing the payload with a direct terminal effect would
not by itself recover them.

A counted L13-only sweep tested exactly nine configurations: three already
declared direction methods by three threshold policies. All three methods had
the same AUC and the same decisions for a given policy:

| Threshold policy | Positive fires | Control fires |
|---|---:|---:|
| balanced binary | 5/6 | 1/6 |
| zero-false-positive training ceiling | 6/6 | 5/6 |
| strict separation gap | 3/6 | 0/6 |

There is no hidden method or threshold in this family that supplies both
coverage and zero observed false activation.

The failure is more specific than absence of relation information. Within
every matched construction, the relation-present minus relation-absent score
is positive for both entities. The six entity-present deltas range from
`+1.814410` to `+3.152150`; the other-entity deltas range from `+1.863347` to
`+3.230877`. Construction-specific offsets, not the within-construction
relation contrast, break the single global threshold. Projecting out the
training construction-mean subspace at ranks 1 through 4 did not improve
coverage: every rank remained at 3/6 positives and 0/6 controls.

Relation score and payload contribution are also coupled rather than
independent on the six positives: Spearman correlation is about `0.714286`.
This supports a shared surface-sensitivity diagnosis.

- Fixed-candidate report SHA-256:
  `548abb5e268dbf92dc01f315fb043d5b9d24e054448c7f6a8e0fd154d663e476`.
- Post-mortem summary SHA-256:
  `36918cad97686a9e75bcff1bf0d3dc404325626903b30fcea622d6178f88d608`.

## Architectural consequence

The current chain conflates two independent jobs:

```text
context eligibility -> L31 FFN payload -> first target ID -> ActiveSpan
```

The L31 executor and the absolute L13 detector score both vary with linguistic
construction. The observed payload misses have contribution norms of about
`0.17` and `1.24`, while successful forms range from about `13.9` to `151.7`;
the safe detector abstains on those same two prompts.

Decoupling eligibility from effect remains necessary:

```text
exact entity alias AND learned relation -> direct terminal residual effect
                                      -> first target ID -> ActiveSpan
```

In that design the detector decision, not incidental activation of a
trace-selected FFN row, is the sole trigger for the already-bounded terminal
effect. The patch still fails closed through `compiled_reject`, `no_fire`, and
`abstain`, and `ActiveSpan` still owns exact continuation and release.

It is not sufficient. A research successor also needs a relation decision
that removes construction offset. The post-mortem identifies two honest
branches:

- a product path compiles exact supported relation surfaces, combines them
  with exact entity aliases, and directly gates the terminal effect;
- a model-bound research path learns from substantially more construction
  groups or evaluates a two-pass matched counterfactual score. The latter has
  a perfect paired ordering on these six spent groups, but doubles inference
  and requires an independently valid rule for producing the relation-absent
  counterfactual.

This consequence is an architecture proposal, not a v1 rescue. Testing it
would require a new fact, new development/calibration/holdout material, a new
ABI and protocol commit, and a separately frozen claim. The current material
cannot be reused for prospective evidence.
