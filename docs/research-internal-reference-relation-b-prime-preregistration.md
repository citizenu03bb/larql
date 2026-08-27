# Internal-reference relation contrast B-prime: post-closure gate

**Status:** frozen before the single replay

**Evidence class:** architecture orientation on the 24 already-spent
factorized-relation-v1 development prompts

**Model boundary:** one exact Gemma 4 E4B native-Q4K vindex

## Question and evidence boundary

Does the L13 relation signal fail under an absolute score mainly because each
linguistic construction adds a prompt-wide offset which can be removed by a
fixed, oracle-free reference from the same forward pass?

This is the B-prime architecture gate. It is not a prospective replication,
calibration, certification, payload rescue, or holdout evaluation. The
factorized-relation-v1 verdict remains a scientific failure at its natural
payload gate. Its calibration and holdout files remain unopened.

The replay uses exactly the same 24 spent development prompts, exact source
patch, model artifact, tokenizer, and capture binary as v1. Its sole purpose is
to record every prompt position at L13; the earlier capture retained only the
last position. No new construction, fact, label, payload, threshold, adapter,
or ABI is introduced.

## Frozen inputs

- Base artifact:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Development suite:
  [factorized-relation-v1/development.json](research-data/factorized-relation-v1/development.json),
  SHA-256
  `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243`.
- Frozen source patch SHA-256:
  `2c69ccd022cb2c24d332eb1acf0f92e5c4a7fb12ec09bd21732de27bea8433fd`.
- Frozen last-position baseline capture SHA-256:
  `4b0a57229b8507cb0b15f83ef4c66539141d8e85fb9170e2daf9d7e3ed2822e3`.
- Capture binary SHA-256:
  `13a5e5076c0888c95d282f745a92b569947e0722a30c4dcd36b0d14ac9e58929`.
- Layer: `13` only.
- Direction method: `raw_conjunction` only.
- Grouping: six construction folds, each containing cells `11`, `10`, `01`,
  and `00`.

## Infrastructure parity gate

Before any B-prime score is interpreted, the replay's L13 vector at the final
prompt position must be element-for-element identical to the archived v1
vector for all 24 rows, matched by fold, factorial cell, name, and complete
prompt. Failure is `infrastructure-inconclusive`; no reference result is read
and no retry is authorized by this protocol.

## Fixed reference family

For each prompt, let `x_last` be the L13 residual at the final token position.
Exactly four reference statistics are admitted, in this priority order:

1. `previous`: the residual at the penultimate prompt position;
2. `mean_body`: the mean residual over positions after BOS and before the
   final position;
3. `first_body`: the residual at position one, immediately after BOS;
4. `entity_span`: the mean residual over the first exact contiguous occurrence
   of either frozen `Archive Zephyr` token sequence,
   `[39140, 7733, 109999]` or `[40187, 7733, 109999]`.

These definitions are label-independent and require no relation-token mask,
surface counterfactual, generated prompt, semantic parser, or fitted anchor.
An absent or ambiguous implementation input is an analysis error, not a cue to
add a fifth reference.

No token ablation, token search, reference weighting, layer search, pooling
variant, rank, nonlinear model, or reference-list extension is permitted after
the replay begins.

## Fixed grouped analysis

For each held-out construction fold:

1. fit the existing `raw_conjunction` unit direction on the final-position
   residuals from the other five folds and all four factorial cells;
2. for each held-out row and each fixed reference, compute
   `d = dot(x_last - x_ref, direction)`;
3. retain the held-out scores without fitting a center-dependent correction or
   any threshold.

For each reference, pool only the 24 out-of-fold scores and compute:

```text
global_gap = min(d for cells 11 and 01) - max(d for cells 10 and 00)
eligible_gap = min(d for cell 11) - max(d for cell 10)
```

The primary criterion is `global_gap > 0`. `eligible_gap` is descriptive only.
Every reference and every row is reported. If more than one reference passes,
the first in the frozen priority order is the sole candidate eligible for a
future prospective protocol. There is no accuracy, AUC, p-value, threshold,
or best operating point in this gate.

## Falsifier and stop rule

- If at least one fixed reference has a strictly positive `global_gap`,
  B-prime survives as an architecture hypothesis only.
- If none does, the additive internal-offset hypothesis is closed for this
  artifact and reference family. The next recommended path is the compiled
  product branch, not a higher-capacity learned detector trained on these six
  groups.
- A zero gap fails. A near miss fails. No threshold, extra reference, prompt,
  layer, or rerun may rescue the result.

A surviving B-prime result does not authorize effecter changes. It authorizes
designing a separate prospective experiment with new material. That protocol
must keep shakeout material disjoint and freeze its claim at the available
sample size. For a frequentist statement no stronger than an observed false
activation rate bounded by about 5% at 95% confidence, the planning floor is
60 independent entity-present control observations, approximately 30 new
construction groups, spread across at least two new facts and three relation
surface families. Ten to twelve additional shakeout constructions remain
outside the reported evaluation. No result may be described as universally or
literally zero-error certified.

## Single replay and resources

The only permitted full-model action is one activation-only replay of the
spent development suite with `--residual-layer 13 --position-residuals`, using
the already-frozen source patch. It runs sequentially in one foreground
`systemd --user` unit with:

- `MemoryHigh=22G`;
- `MemoryMax=24G`;
- `MemorySwapMax=8G`;
- `OOMPolicy=stop`;
- `RuntimeMaxSec=1800`.

The real process list must be checked first. A resident E4B process forbids the
launch. The capture, manifest, raw report, and journal remain in the private
research directory. The public repository receives the protocol, analyzer,
tests, and a bounded result document.

No calibration capture, holdout access, patch compilation, terminal adapter,
generation, ABI change, or push is part of this gate.
