# Internal-reference relation contrast B-prime v2: result

**Verdict:** the three executable reference arms failed; no B-prime candidate
nominated

**Frozen source:** `115f513b9e2b3cc9a98b5ffd9cd1b156586d867d`

**Evidence label:** exploratory

**Holdout or calibration accessed:** no

## Result

The corrected v2 analyzer reused the exact 24-row all-position L13 capture and
verified element-for-element parity with the archived final-position capture.
It evaluated three label-independent references while retaining the relation
direction learned on `x_last` from the other five construction groups:

| Reference | Global gap | Eligible `11` vs `10` gap | Pass |
|---|---:|---:|---:|
| `previous` | `-5.368470` | `-5.368470` | no |
| `mean_body` | `-1.114034` | `-1.114034` | no |
| `first_body` | `-1.017260` | `-0.967465` | no |

The frozen criterion required a strictly positive global gap. Zero and near
misses failed, so no reference was selected and no threshold was fitted.

This falsifies these three reference functions under the v2 estimator. It does
not falsify all internal references, a direction learned directly in a
differential space, or the existence of construction-local relation ordering.
The effective inter-construction sample size remains six.

## Evidence boundary

- Position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`.
- Analysis SHA-256:
  `88bd55872c6141a6eecc6aefd3f06f07acac1881c6b28e07e3999c839a757527`.
- V2 analysis manifest SHA-256:
  `af846bc14fd6f43e28fbd892d0775e9357917dfb52d08b8592bc98d8f81881d1`.
- No payload, detector artifact, terminal adapter, ABI, calibration, or holdout
  was produced or accessed.

## Unexecuted fourth arm

The original `entity_span` arm raised an exception before any v1 contrast
report because its alias table covered `Archive Zephyr` but not the factorial
control entity `Archive Nimbus`. V2 removed it before reading the three scores.

Under the progressive evidence protocol, completing that exact predeclared arm
with a total alias table is a mechanical completion, not evidence for the arm.
Changing the direction to fit `x_last - x_ref` would instead be a new estimator
created after the observed failures. The two actions are therefore separated.
