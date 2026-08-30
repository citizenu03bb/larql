# Counterfactual entity-swap interaction: exploratory nomination gate

**Status:** closed; see
[the result](research-counterfactual-entity-swap-interaction-results.md)

**Evidence label:** `exploratory`; architecture orientation on spent material

**Model boundary:** one exact Gemma 4 E4B native-Q4K vindex

## Question and distinction from the closed depth atlas

The closed full-depth atlas tested absolute single-pass scores at the final
prompt position. L11 was its only individually separating layer, but no stable
three-layer band survived and no layer was nominated. That verdict remains
binding; this experiment does not relax its nomination rule or promote L11.

The spent factorial captures expose a structurally different observable. Each
construction contains a target-entity prompt and a fixed reference-entity
prompt under both relation-present and relation-absent predicates:

```text
z_present = h(11) - h(01)
z_absent  = h(10) - h(00)
```

Here `h` is the raw pre-FFN residual at the final prompt position. The target
entity is `Archive Zephyr`; the fixed reference is `Archive Nimbus`. The new
question is:

> Does a leave-one-construction-out linear readout of the entity-by-relation
> interaction remain globally separated after the within-construction entity
> swap cancels an additive construction offset?

This approximates a possible two-pass runtime detector: score the original
prompt and a deterministic prompt in which the compiled entity span is replaced
by a fixed reference entity, then compare their residuals. It is not a second
attempt at an absolute single-pass threshold.

A positive result may nominate one layer for exactly two fresh adversarial
shakeout construction groups under a separate protocol. It cannot establish
semantic understanding, a deployable detector, calibration, certification,
false-positive rate, or runtime acceptability.

## Frozen inputs and known approximation

- Existing private capture:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/full-depth-relation-atlas/capture-20260829.json`.
- Capture SHA-256:
  `d4f9170e17e3ec6c13725776ffc2dda761861d7fdd601b9d46a891d5ed4036f1`.
- Capture source commit:
  `c280f497a41737e059a596323003991f559a13d8`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Spent suite:
  [factorized-relation-v1/development.json](research-data/factorized-relation-v1/development.json),
  SHA-256
  `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243`.
- Rows: 24, six construction folds, one each of cells `11/10/01/00`.
- Layers: exactly L0 through L41; configuration count 42.
- Position and representation: final prompt position, raw pre-FFN residual.
- Target entity: `Archive Zephyr`.
- Fixed reference entity: `Archive Nimbus`.
- Grouping: leave one complete construction fold out.

Within each fold, `11` and `01` share relation and surface construction while
changing only the entity string; the same is true for `10` and `00`. The
reference entity is therefore fixed and oracle-free with respect to the label.

The tokenized reference prompt is one token shorter than its target-entity
counterpart in every fold. The difference observable may consequently include
an entity-length or positional component. The factorial difference-of-
differences can cancel such a component only if it is sufficiently additive.
No same-length alias, retokenization, alignment, or new capture is introduced
after observing the result.

The factorized-relation calibration and holdout suites remain unopened. Alder
and Alder-v2 holdouts remain sealed. Product Minas Tirith material cannot enter
the analysis. No E4B forward pass, threshold fitting, payload, adapter, ABI, or
runtime detector is part of this gate.

## Frozen paired analysis

For every layer, create one relation-present and one relation-absent entity-swap
vector per construction. For each held-out fold `c`, using only the other five
folds:

```text
z1_f       = h_f(11) - h_f(01)
z0_f       = h_f(10) - h_f(00)
center_-c  = mean({z1_f, z0_f | f != c})
direction_-c = unit(mean(z1_f - center_-c | f != c)
                  - mean(z0_f - center_-c | f != c))
score(z)   = dot(z - center_-c, direction_-c)
```

This yields twelve out-of-fold scores: six relation-present swap scores and six
relation-absent swap scores. Report:

```text
global_gap = min(score(z1)) - max(score(z0))
paired_delta_f = score(z1_f) - score(z0_f)
minimum_paired_delta = min_f(paired_delta_f)
margin_ratio = global_gap / minimum_paired_delta
```

The ratio is defined only when `minimum_paired_delta > 0`. It makes the gap
floor relative to the weakest within-construction interaction effect; it is a
frozen project convention, not a statistical confidence bound.

## Frozen continuity and selection rules

For every adjacent layer pair and every held-out fold, compute the cosine
between the two fold-trained interaction directions. For candidate layer `L`,
each available immediate neighbor receives:

```text
adjacent_min_cosine(L, neighbor)
    = min over six corresponding held-out folds of cosine(d_L,-c, d_neighbor,-c)
```

The layer's continuity statistic is the larger adjacent minimum. At L0 and L41
only the single available neighbor is used. Continuity passes at `>= 0.7`.
There are 41 reported adjacent comparisons; their threshold is not tuned.

A layer survives only if all conditions hold:

- all six paired deltas are strictly positive;
- `global_gap > 0`;
- `margin_ratio >= 0.5`;
- best adjacent minimum direction cosine `>= 0.7`;
- all values are finite and all twelve held-out scores are present exactly once.

The `0.5` ratio and `0.7` cosine floors are preregistered conventions. They do
not estimate a false-positive rate.

If multiple layers survive, nominate exactly one: the layer with the largest
`margin_ratio`; an exact tie selects the lower layer index. All 42 layer reports
and all 41 adjacent comparisons are emitted before this deterministic rule is
applied. No raw gap is maximized across depth.

## Stop rule and next authorization

If no layer survives, close semantic detector work on this artifact under the
available spent material. Do not add another reference entity, layer ensemble,
position, pooling, normalization, ratio floor, cosine floor, nonlinear head,
or E4B capture after reading the result.

If one layer is nominated, the only authorization is to write a separately
frozen two-group shakeout protocol. The two groups must be fixed before capture:
one relation-adjacent control predicate and one nominalized or long-distance
positive construction. The layer, entity swap, direction fit, ratio floor, and
stop rule remain fixed; there is no layer resweep. The two groups are a
falsification shakeout, not certification. A larger prospective claim would
still require independently justified material and calibration.

No `ContextDetectorSpec`, two-pass runtime, threshold, payload, or patch ABI is
implemented by this analysis. The public verdict is either
`exploratory two-pass candidate nominated`, `exploratory candidate rejected`,
or `analysis-inconclusive`.
