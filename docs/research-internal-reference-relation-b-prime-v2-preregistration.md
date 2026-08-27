# Internal-reference relation contrast B-prime v2: corrected analysis gate

**Status:** frozen before reading the captured residual contrasts

**Predecessor:** B-prime v1 closed as infrastructure-inconclusive

**Evidence class:** architecture orientation on spent prompts

## Correction boundary

The v1 replay already produced one all-position L13 capture. Its analyzer
verified exact final-position parity and then raised an exception because the
fourth reference required an `Archive Zephyr` alias in rows whose entity was
`Archive Nimbus`. It wrote no report, and no residual contrast was manually
inspected.

V2 changes exactly one degree of freedom before reading those values: it
deletes the non-total `entity_span` reference. It neither replaces it nor adds
another reference. The capture is reused byte-for-byte; there is no second
E4B invocation.

## Frozen inputs and analysis

- Position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`.
- Baseline last-position capture SHA-256:
  `4b0a57229b8507cb0b15f83ef4c66539141d8e85fb9170e2daf9d7e3ed2822e3`.
- Layer: `13` only.
- Direction: `raw_conjunction`, fit leave-one-construction-out on final
  positions from the other five folds and all factorial cells.
- Scores: `d = dot(x_last - x_ref, direction)` on each held-out row.
- No threshold, center correction, layer choice, fitted anchor, rank, AUC,
  p-value, or operating-point search.

Exactly three references remain, in fixed priority order:

1. `previous`: penultimate prompt position;
2. `mean_body`: mean of positions after BOS and before the final position;
3. `first_body`: position one, immediately after BOS.

Each is defined from position alone for every one of the 24 prompts and is
independent of the cell label and relation-token identity.

For each reference:

```text
global_gap = min(d for cells 11 and 01) - max(d for cells 10 and 00)
eligible_gap = min(d for cell 11) - max(d for cell 10)
```

The sole survival criterion is `global_gap > 0`. The first passing reference
in frozen priority order becomes the sole candidate that may be carried into
a future prospective protocol. `eligible_gap` is descriptive only.

## Stops

- Final-position parity is checked again before scoring. A mismatch remains
  infrastructure-inconclusive.
- If no reference has `global_gap > 0`, B-prime closes for this artifact and
  reference family. Zero and near-zero fail.
- No fourth reference, threshold, prompt, layer, or model replay is permitted.
- A pass is architecture orientation only. It does not calibrate or certify a
  detector and does not authorize holdout access or effecter work.
- Any prospective successor needs independently frozen new facts,
  constructions, shakeout material, calibration, and claim-sized evaluation.
