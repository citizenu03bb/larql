# Entity-span internal-reference arm: mechanical completion

**Status:** frozen before the first executable score read

**Evidence label:** exploratory completion of one predeclared arm

**Parent lineage:** B-prime v1 and v2; their verdicts remain unchanged

## Why this completion is permitted

`entity_span` was fourth in the B-prime v1 frozen reference list. It did not
produce a statistic because the implementation recognized only `Archive
Zephyr`; factorial rows containing `Archive Nimbus` raised an exception. The
failure is visible from prompt token metadata without inspecting entity-span
residual scores.

This protocol changes only alias-table completeness. It does not add a fifth
reference, relearn the direction in a differential space, normalize a score,
fit a threshold, change a layer, or replay the model.

## Frozen inputs

- All-position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`.
- Baseline final-position capture SHA-256:
  `4b0a57229b8507cb0b15f83ef4c66539141d8e85fb9170e2daf9d7e3ed2822e3`.
- Layer: `13`.
- Six construction folds, each with cells `11`, `10`, `01`, and `00`.
- Direction estimator: v2 `raw_conjunction` fit on `x_last` from the other five
  folds.
- Reference: mean L13 residual over the first exact supported entity span in
  the same prompt.

The complete fixed aliases are:

- `Archive Zephyr` at prompt start: `[39140, 7733, 109999]`;
- ` Archive Zephyr` after a space: `[40187, 7733, 109999]`;
- `Archive Nimbus` at prompt start: `[39140, 129201]`;
- ` Archive Nimbus` after a space: `[40187, 129201]`.

The alias used is determined solely by exact contiguous prompt IDs. Cell labels
do not select the span.

## Fixed analysis

For each held-out construction `c`:

```text
d_-c = raw_conjunction_direction(x_last on the other five constructions)
x_ent = mean(L13 residual over the exact entity token span)
s_i = dot(x_last_i - x_ent_i, d_-c)
```

No center term remains in the within-prompt difference. No direction is learned
from `x_last - x_ent`.

The primary metric is:

```text
global_gap = min(s for cells 11 and 01) - max(s for cells 10 and 00)
```

Survival requires `global_gap > 0`. Secondary diagnostics, unable to nominate
or rescue the arm, are:

- `eligible_gap = min(s for 11) - max(s for 10)`;
- the minimum paired present-minus-absent delta across both entities and all
  six constructions;
- the paired present-minus-absent projection carried by `x_ent`, reported to
  show whether the reference itself contains relation information.

## Falsifier and stop rule

- `global_gap <= 0` closes the `entity_span` arm and the four-reference B-prime
  family under the original last-position direction.
- A positive gap nominates, but does not validate, the completed reference for
  a fresh prospective protocol.
- Exactly one analysis execution and one report are permitted.
- No refit on the differential vectors, fifth reference, normalization,
  threshold, layer, position atlas, or model replay may be added to this read.
- A new differential-direction estimator is a separate architecture. It may be
  nominated for fresh material but cannot complete or rescue this lineage.

No E4B process, calibration file, holdout, payload, terminal adapter, ABI
change, or push is authorized by this protocol.
