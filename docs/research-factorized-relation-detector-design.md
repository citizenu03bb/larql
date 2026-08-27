# Factorized relation-only contextual detector design

**Status:** architecture design on fully spent Alder-v2 development captures;
the first prospective v1 closed before detector fitting at its independent
payload gate

**Evidence boundary:** exploratory only; not calibration or certification

**Model boundary:** exact Gemma 4 E4B Q4K artifact used by Alder-v2

## Decision

The next prospective detector candidate separates the two contextual factors:

- entity presence is compiled as an exact token-ID condition over a frozen
  entity and frozen aliases;
- relation eligibility alone is learned from model residuals;
- the fixed exploratory candidate reads the last prompt position at layer 13;
- the runtime decision is ternary: `fire`, `abstain`, or `no_fire`;
- `abstain` and `no_fire` both keep the payload inactive, but remain distinct
  in evidence and certification reports.

This is not Alder-v3. Alder-v2 is closed, its holdout remains sealed, and all
48 Alder-v2 development captures are spent architecture-design material. The
candidate may become evidence only on a new fact under a separately frozen
protocol.

The first such protocol is now also closed. Its trace-guided L31 payload
selected the new first target ID on only 4/6 development positives, so its
frozen stop rule prevented relation fitting and calibration. See
[factorized-relation-v1 results](research-factorized-relation-v1-results.md).
That result does not test the L13 detector candidate; it instead motivates a
future executor directly triggered by the detector decision rather than by an
incidental FFN-row activation.

## Why the factors are asymmetric

The failed Alder-v2 row had a strongly present entity margin and a failing
relation margin:

```text
Project Alder is kept by
entity = +0.552672
relation = -0.099888
```

Learning the entity direction therefore spends scarce examples on a condition
that can be checked exactly for the supported surface. Compiling that condition
also makes the safety contract explicit: aliases and coreference not listed in
the patch are unsupported and must abstain rather than being inferred from an
unvalidated entity probe.

The relation remains model-bound. A lexical relation bank would enumerate
known surfaces and would not establish semantic generalization.

## Spent-data comparison

`analyze_factorized_relation.py` replaces the entity factor with the dataset's
entity bit as an oracle proxy, fits the relation from all four factorial cells,
and evaluates only entity-present `11` positives and `10` controls. The strict
`separation_gap` policy fits two thresholds from each training fold:

- `no_fire_threshold = max(training relation-absent score)`;
- `fire_threshold = min(training relation-present score)`;
- scores strictly between them abstain;
- an overlapping training fold is invalid.

The comparison below is leave-one-entire-suite-out and post-hoc. It is useful
for choosing the next hypothesis only.

| Candidate | Positive fires | Control fires | Abstentions | Coverage |
|---|---:|---:|---:|---:|
| L6 `mean_no_bos` relation-only | 8/12 | 0/12 | 13/24 | 66.7% |
| **L13 `last` relation-only** | **10/12** | **0/12** | **6/24** | **83.3%** |
| L30 `last` relation-only | 8/12 | 0/12 | 6/24 | 66.7% |
| L31 `mean_no_bos` relation-only | 9/12 | 0/12 | 5/24 | 75.0% |

The same L13 candidate reaches 12/12 positives and 0/12 controls under the old
balanced binary threshold, with worst signed margin `+0.039525`. That result is
not the selected metric: the stricter ternary evaluation exposes two fragile
prospective positives as abstentions instead of presenting `24/24` as robust
classification.

A threshold placed just above the training negative ceiling is also rejected.
It fired on all four held-out dev4 `10` controls. Zero observed false positives
inside the fitting suites is not precision certification.

The payload-local hypothesis is not supported by these captures. The L30/L31
candidates retain the `kept by` failure and cover less than L13 under the same
ternary contract. L13 is a development-selected hypothesis, not a causal claim
that layer 13 universally contains relation semantics.

## Prospective architecture

The candidate wire semantics are:

```text
entity_match = any frozen token-ID alias occurs in the prompt
relation_score = dot(last_residual_L13 - center, relation_direction)

if not entity_match:
    compiled_reject
elif relation_score >= fire_threshold:
    fire
elif relation_score <= no_fire_threshold:
    no_fire
else:
    abstain
```

Only `fire` makes the contextual payload and terminal residual adapter visible.
Every other state is fail-closed. The inference evidence must preserve the
state, raw score, both thresholds, matched entity alias, and payload slot.

The current `.vlp` format cannot express this contract: it supports only
`mean_no_bos`, two learned directions, and one threshold per factor. No format
or runtime extension is authorized by the Alder exploration. The ABI changes
only after the new fact passes development and calibration gates outside the
runtime.

## Draft prospective protocol

This section defines the experiment shape. It is not frozen until the exact
new fact, prompts, aliases, target IDs, thresholds, checksums, and stop rules
are written in a separate preregistration and committed before capture.

### Fixed candidate

- detector layer: 13;
- representation: last prompt position;
- relation method: raw factorial conjunction direction;
- entity method: exact token-ID aliases, frozen before capture;
- threshold method: strict calibration separation gap;
- payload: one natural first token plus the already-fixed softcap-aware
  terminal residual adapter and `ActiveSpan` continuation;
- no layer, representation, method, rank, threshold-policy, or payload sweep.

### Material split

- development: at least six construction groups, each containing matched
  `11/10/01/00` cells;
- calibration: at least two disjoint construction groups with the same cells;
- sealed holdout: at least six disjoint construction groups, including one
  construction family absent from Alder and from the new development set;
- the grouping variable is the linguistic construction, not an arbitrary file
  or row split;
- unsupported alias-only or coreference-only forms are declared abstention
  cases before capture, not relabelled after results.

The relation direction is fitted once on development. Calibration alone sets
the two thresholds. A non-positive calibration separation gap returns
`UncompilableEdit` and keeps the holdout sealed.

### Holdout acceptance

The holdout is opened once only after the detector artifact, entity aliases,
payload, terminal adapter, ABI, checksums, and evidence schema are frozen.
Acceptance requires all of the following:

- zero `fire` decisions on every `10` control;
- every `01` and `00` row rejected by the compiled entity condition;
- at least five of six supported `11` construction groups fire;
- every uncovered supported positive abstains rather than returning
  `no_fire`;
- minimum normalized fire and control-safety margins of at least `0.10`, where
  the normalization denominator is the frozen calibration separation gap;
- every fired positive selects the first target ID naturally with margin at
  least `+0.5`, emits the exact target span, arms once, and releases;
- every non-fired row retains the complete base path without payload or
  terminal-adapter application.

Any failure closes this candidate. It does not authorize a different layer,
new alias, threshold movement, binary fallback, extra adapter, or use of the
Alder holdout.

## Stop rules

- Alder-v2 data may reproduce this design report but may not calibrate or
  certify the prospective detector.
- The next fact is selected once. A failed candidate is not replaced within
  the same protocol.
- A new E4B process is not launched until the exact preregistration and source
  are committed and a provenance manifest is written.
- Full-vindex processes remain sequential and bounded by the established
  `MemoryHigh=22G`, `MemoryMax=24G`, and `MemorySwapMax=8G` envelope.
- A result may be pass, scientific failure, or infrastructure-inconclusive.
  "Almost" and aggregate binary accuracy are not outcomes.

## Reproduction of the exploratory comparison

Run the tool once per already-chosen candidate. For example:

```bash
python3 crates/larql-lql/examples/analyze_factorized_relation.py \
  dev1=/path/to/captures-dev1.json \
  dev2=/path/to/captures-dev2.json \
  dev3=/path/to/captures-dev3.json \
  dev4=/path/to/captures-dev4.json \
  --layer 13 \
  --representation last \
  --method raw_conjunction \
  --threshold-policy separation_gap
```

The output labels itself exploratory and reports each suite independently.
