# Full-depth relation atlas: exploratory emergence map

**Status:** preregistered; capture not yet run

**Evidence label:** `exploratory`; architecture orientation on spent material

**Model boundary:** one exact Gemma 4 E4B native-Q4K vindex

## Question and distinction from the closed atlas

The closed late-decision atlas tested exactly L29, L31, and L33 and rejected
all three under its frozen rule. Its stop rule remains binding: this experiment
does not add a fourth late-layer candidate or reinterpret that verdict.

This new experiment asks a broader family-level question requested after that
closure:

> Does any stable contiguous depth band expose a relation-only scalar at the
> final prompt position that transfers across the six spent construction
> groups once entity eligibility is compiled?

The motivation is diagnostic rather than tuning. Transformer representations
change continuously with depth; if a globally thresholdable linear relation
factor emerges, it should persist across more than one adjacent block. A
single isolated positive among 42 inspected layers is therefore reported but
cannot nominate an architecture.

A positive result may nominate one layer for a fresh prospective protocol. It
cannot establish that semantics "lives" in that layer, semantic understanding,
calibration, deployable thresholds, prospective generalization, certification,
or a false-positive rate. A negative result rejects only the tested family:
one-pass, final-position, raw-linear, single-layer relation scoring on this
artifact and material.

## Frozen inputs and exclusions

- Base artifact:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Model depth: 42 transformer layers from `index.json`.
- Spent development suite:
  [factorized-relation-v1/development.json](research-data/factorized-relation-v1/development.json),
  SHA-256
  `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243`.
- Frozen source patch SHA-256:
  `2c69ccd022cb2c24d332eb1acf0f92e5c4a7fb12ec09bd21732de27bea8433fd`.
- Rows: exactly 24, partitioned into six construction folds with one each of
  cells `11`, `10`, `01`, and `00`.
- Layers: exactly every layer from `L0` through `L41`, configuration count 42.
- Position: final encoded prompt position only.
- Representation: raw pre-FFN residual at that layer and position.
- Direction: `raw_conjunction` relation-only direction.
- Grouping: leave one complete construction fold out.

The factorized-relation calibration and holdout suites remain unopened. Alder
and Alder-v2 holdouts remain sealed. Product Minas Tirith material cannot enter
the fit or metrics. No alternative position, pooling, normalization, internal
reference, direction estimator, threshold search, nonlinear head, payload,
adapter, ABI, or runtime detector is part of this experiment.

## Capture contract

The existing activation-only suite runner is used unchanged. One invocation
must retain all prompt positions at exactly L0 through L41 and write one
private capture. A requested layer that is missing, duplicated, non-finite, or
outside the model is an infrastructure error.

Before the replay:

1. this protocol, analyzer, and deterministic tests are committed;
2. the release example binary is rebuilt from that commit;
3. a private manifest records the source commit and SHA-256 hashes of this
   protocol, suite, patch, binary, vindex manifests, tokenizer, output path,
   and unit name;
4. the real process list is checked and no E4B process may be resident.

The sole full-model command has this logical shape, with one repeated
`--residual-layer N` argument for each integer `N` in `0..41`:

```bash
multitoken_insert_baseline \
  /mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex \
  --suite docs/research-data/factorized-relation-v1/development.json \
  --apply-patch /private/factorized-relation-v1-source.vlp \
  --activation-only \
  --position-residuals \
  --residual-layer 0 ... --residual-layer 41 \
  --residual-out /private/full-depth-relation-atlas-capture.json
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

Only the 24 out-of-fold scores are pooled. For each of the 42 layers report:

```text
global_gap = min(score for relation-present cells 11/01)
           - max(score for relation-absent cells 10/00)

eligible_gap = min(score for cell 11) - max(score for cell 10)
```

For every fold and entity bit, also report the paired relation-present minus
relation-absent delta. Direction cosines and score ranges are descriptive only.
No threshold is fit: `global_gap > 0` is the existence test for some separating
threshold on the spent out-of-fold scores.

A layer passes individually only if:

- `global_gap > 0`;
- every one of the twelve paired deltas is strictly positive;
- all values are finite and every held-out row appears exactly once.

The primary map reports every individually passing layer and every maximal
contiguous run. An architecture is nominated only if at least three adjacent
layers pass. If that occurs, nominate the middle layer of the earliest
three-layer passing window: for a first window `Lk, L(k+1), L(k+2)`, nominate
`L(k+1)`. This fixed rule searches for the earliest stable emergence and does
not maximize a post-hoc score whose scale may change with depth.

An isolated passing layer or a run of length two is diagnostic only and yields
`nominated_layer=null`. The analyzer always reports all 42 layers before
applying the rule.

## Stop rule and next authorization

If no three-layer passing window exists, close the full-depth raw-linear atlas
with `no architecture justified`. After reading the map, do not add another
representation, position, estimator, smoothing rule, layer ensemble, threshold,
or nonlinear model to this evidence chain. No detector ABI work is authorized.

If a stable band survives, the only authorization is to write a separately
frozen prospective protocol for the nominated layer. Thresholds and any
ternary `fire | abstain | no_fire` boundaries must use fresh disjoint material,
never these six spent groups.

The final public result is one of:

- `exploratory stable band nominated`;
- `exploratory map positive but no stable band`;
- `exploratory candidate rejected`;
- `infrastructure-inconclusive`.

A near miss does not alter the rule.
