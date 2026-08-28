# Late-decision relation atlas: exploratory nomination gate

**Status:** closed; see
[the result](research-late-decision-relation-atlas-results.md)

**Evidence label:** `exploratory`; architecture orientation on spent material

**Model boundary:** one exact Gemma 4 E4B native-Q4K vindex

## Question and mechanism

The closed L13 experiments preserved positive relation-present-minus-absent
ordering inside every matched construction, but no tested absolute or internal
contrast score produced a strictly positive global gap across constructions.
This experiment asks one structurally different question:

> Does the final prompt position at the L31 effect site, or one symmetric
> flanking layer, contain a relation-only scalar that transfers across the six
> spent construction groups once entity eligibility is treated as compiled?

The mechanistic primary layer is L31 because the bounded terminal effect is
already expressed in that late decision region. L29 and L33 are fixed flanking
diagnostics, not an open layer sweep. Entity eligibility is not learned here;
the future architecture would compile exact entity aliases and learn only the
relation factor.

A positive result may nominate one score function for fresh grouped material.
It cannot establish semantic understanding, calibration, deployable thresholds,
prospective generalization, certification, or a false-positive rate.

## Frozen inputs and exclusions

- Base artifact:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Spent development suite:
  [factorized-relation-v1/development.json](research-data/factorized-relation-v1/development.json),
  SHA-256
  `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243`.
- Frozen source patch SHA-256:
  `2c69ccd022cb2c24d332eb1acf0f92e5c4a7fb12ec09bd21732de27bea8433fd`.
- Rows: exactly 24, partitioned into six construction folds with one each of
  cells `11`, `10`, `01`, and `00`.
- Layers, in nomination priority order: `L31`, `L29`, `L33`.
- Position: final encoded prompt position only.
- Representation: raw pre-FFN residual at that layer and position.
- Direction: `raw_conjunction` relation-only direction.
- Grouping: leave one complete construction fold out.

The factorized-relation calibration and holdout suites remain unopened. Alder
and Alder-v2 holdouts remain sealed. No product Minas Tirith material may enter
the fit or the metrics. No payload, terminal adapter, threshold, ABI, or runtime
detector is compiled by this gate.

## Capture contract

The existing activation-only suite runner is extended mechanically so
`--residual-layer` may be repeated. One invocation must retain all positions at
exactly L29, L31, and L33 and write one private capture. A requested layer that
is missing, duplicated, non-finite, or outside the model is an infrastructure
error.

Before the replay:

1. the protocol, capture filtering, analyzer, and deterministic tests are
   committed;
2. the release example binary is rebuilt from that commit;
3. a private manifest records the source commit and SHA-256 hashes of this
   protocol, suite, patch, binary, vindex manifest, tokenizer, output path, and
   unit name;
4. the real process list is checked and no E4B process may be resident.

The sole full-model command has this logical shape:

```bash
multitoken_insert_baseline \
  /mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex \
  --suite docs/research-data/factorized-relation-v1/development.json \
  --apply-patch /private/factorized-relation-v1-source.vlp \
  --activation-only \
  --position-residuals \
  --residual-layer 29 \
  --residual-layer 31 \
  --residual-layer 33 \
  --residual-out /private/late-decision-relation-atlas-capture.json
```

It runs once, sequentially, in a foreground `systemd --user` unit with
`--wait --pipe --collect`, `MemoryHigh=22G`, `MemoryMax=24G`,
`MemorySwapMax=8G`, `OOMPolicy=stop`, and `RuntimeMaxSec=1800`.

## Frozen grouped analysis

For each layer independently and each held-out fold `c`:

```text
center_-c = mean(x on the other five folds)
mu_xy     = mean(x - center_-c for training cell xy)
direction = unit(0.5 * ((mu_11 + mu_01) - (mu_10 + mu_00)))
score(x)  = dot(x - center_-c, direction)
```

Only the 24 out-of-fold scores are pooled. For each layer report:

```text
global_gap = min(score for relation-present cells 11/01)
           - max(score for relation-absent cells 10/00)

eligible_gap = min(score for cell 11) - max(score for cell 10)
```

For every fold and entity bit, also report the paired relation-present minus
relation-absent delta. Direction cosines, score ranges, means, accuracy, and AUC
may be descriptive only.

A layer survives only if:

- `global_gap > 0`;
- every one of the twelve paired deltas is strictly positive;
- all values are finite and every held-out row appears exactly once.

If multiple layers survive, nominate the first in the frozen mechanistic order
`L31`, `L29`, `L33`. Raw gaps from different layers are not compared as though
they necessarily share a calibrated scale. The analyzer reports all three
layers before returning the single nomination or `none`.

## Stop rule and next authorization

If no layer survives, this late-decision raw-linear atlas closes with
`no architecture justified`. After reading results, do not add L28/L30/L32,
another position, pooling, normalization, reference, threshold, nonlinear
head, or fourth configuration. No detector ABI work is authorized.

If one layer survives, the only authorization is to write a separately frozen
nomination and shakeout protocol. A later runtime may represent exact compiled
entity aliases, the frozen `last` relation score, and a ternary
`fire | abstain | no_fire` decision while reusing the terminal effect and
`ActiveSpan`. Thresholds must come from fresh disjoint calibration, never these
six spent groups.

The final public result is either `exploratory candidate nominated`,
`exploratory candidate rejected`, or `infrastructure-inconclusive`. A near miss
does not alter the rule.
