# Differential first-body relation direction: exploratory nomination test

**Status:** closed; see
[the result](research-differential-first-body-direction-results.md)

**Evidence label:** exploratory; new estimator after B-prime

**Model process required:** no

## Question

Did B-prime fail partly because it projected differential vectors onto a
direction learned in the absolute final-position space?

This test learns the relation direction in the same differential space it
scores. It is not a completion or rescue of B-prime. The estimator was proposed
after observing B-prime failures, so even a positive result can only nominate
a candidate for fresh prospective material.

## Fixed reference and mechanism

The sole reference is L13 position one, the first lexical token after BOS:

```text
z_i = x_last_i - x_first_body_i
```

Within every matched construction/entity pair, the first token ID must be
identical between relation-present and relation-absent prompts. Because the
model is causal, its position-one residual cannot contain information from the
later tokens which distinguish the relation. This gives `z` a concrete
interpretation: the state change accumulated from a matched prefix anchor to
the decision position.

No other reference is evaluated. `first_body` was observed in B-prime v2 and
therefore this choice remains post-hoc; its result has no independent evidence
value.

## Frozen analysis

Inputs and grouping remain:

- all-position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`;
- baseline final-position capture SHA-256:
  `4b0a57229b8507cb0b15f83ef4c66539141d8e85fb9170e2daf9d7e3ed2822e3`;
- layer `13`;
- six construction folds, each containing `11`, `10`, `01`, and `00`.

For each held-out construction `c`:

```text
z_i = x_last_i - x_first_body_i
detector_-c = raw_conjunction fit on z from the other five constructions
s_i = dot(z_i - center_-c, direction_-c)
```

The primary metric pools only the 24 out-of-fold scores:

```text
global_gap = min(s for cells 11 and 01) - max(s for cells 10 and 00)
```

Survival requires `global_gap > 0`. Secondary non-nominating diagnostics are:

- `eligible_gap` for `11` versus `10`;
- minimum paired present-minus-absent score delta;
- per-fold paired deltas;
- cosine stability among the six learned differential directions.

There is no threshold, AUC, p-value, normalization, rank, layer, position, or
reference selection.

## Interpretation and stop rule

- `global_gap > 0` nominates this exact estimator for a separately generated,
  fresh, grouped prospective protocol. It does not validate or calibrate it.
- `global_gap <= 0` rejects this exact differential first-body candidate for
  nomination.
- An unmatched first token, parity mismatch, missing position, NaN, or zero
  direction is analysis-inconclusive and does not authorize changing the
  anchor.
- Exactly one execution and one report are permitted.
- Failure does not authorize `mean_body`, `previous`, `entity_span`, a position
  atlas, a learned anchor, or a nonlinear detector in this arc.

Detector/effecter coupling is explicitly outside this test. No model run,
payload, terminal adapter, ABI, calibration, holdout, or push is authorized.
