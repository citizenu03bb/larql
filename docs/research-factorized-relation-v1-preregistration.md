# Factorized relation detector prospective replication v1

**Status:** closed at the natural payload gate; see
[the result](research-factorized-relation-v1-results.md)

**Architecture source:** spent Alder-v2 captures, never Alder holdout evidence

**Model boundary:** one exact Gemma 4 E4B Q4K vindex

## Question and claim boundary

Can the fixed factorized detector architecture selected on spent Alder-v2
development captures condition a new exact multi-token behaviour patch with
zero observed control activations and explicit abstention on unsupported
relation surfaces?

The experiment tests one synthetic fact on one model artifact. A pass would
replicate a bounded mechanism: exact entity matching, learned relation
eligibility, conditional first-token intervention, and exact `ActiveSpan`
continuation. It would not establish arbitrary edit compilation, factual
knowledge acquisition, cross-model transfer, alias discovery, or coreference
resolution.

The association is deliberately fictional:

- entity: `Archive Zephyr`;
- relation class: the person responsible for mapping the archive;
- target: `Nora Veldt`;
- target IDs: `[107545, 782, 7337, 236745]` (`T=4`);
- supported entity surface: `Archive Zephyr` only;
- exact entity alias ID sequences: `[39140, 7733, 109999]` at prompt start
  and `[40187, 7733, 109999]` after a space.

The target and both alias sequences were obtained by tokenizer-only preflight
before any model weight was loaded. They may not be changed or extended after
this document is frozen.

## Fixed artifact and candidate

- Base artifact:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Tokenizer SHA-256:
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
- Detector layer: 13.
- Detector position: last prompt position.
- Relation method: `raw_conjunction`.
- Entity method: exact contiguous match of either frozen token-ID sequence.
- Runtime states: `fire`, `abstain`, `no_fire`, and `compiled_reject`.
- Payload compiler: the existing deterministic `COMPOSE` path, once only.
- First-token repair: the already-fixed softcap-aware terminal residual
  compiler, once only and only after detector calibration passes.
- Continuation: exact four-ID `ActiveSpan`, armed once and released after the
  fourth target ID.

Layer 13, the last-position representation, and the raw factorial relation
direction are post-hoc choices from spent Alder-v2 development data. This
protocol gives them one prospective opportunity. There is no layer,
representation, pooling, direction, threshold-policy, rank, payload-row, or
adapter sweep.

## Frozen material and grouping

The exact public suite files are:

- [development](research-data/factorized-relation-v1/development.json): six
  construction groups, 6 `11` positives and 18 matched controls; SHA-256
  `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243`;
- [calibration](research-data/factorized-relation-v1/calibration.json): two
  disjoint construction groups, 2 positives and 6 matched controls; SHA-256
  `0a3ab50667b4d9612c9654437fadab54e9ac4754b491540af9670f08b351a1df`;
- [sealed holdout](research-data/factorized-relation-v1/holdout.json): six
  disjoint construction groups, 6 positives and 18 matched controls; SHA-256
  `8f5f774c6a7a296ba11408044fbcd8bdba99d365402a581e5011f68bebd8d5ed`.

Each group contains one row from every factorial cell:

- `11`: supported entity and relation;
- `10`: supported entity, different relation;
- `01`: different entity, relation present;
- `00`: different entity and different relation.

The positive prompt at index `i` and the three controls whose `fold` is `i`
form one construction group. Group identity is the construction, never the
file or an individual row. Calibration and holdout relation wording denotes
the same frozen relation class; those files must be evaluated by applying the
single patch compiled from development, never by compiling a new patch from
their `relation` field.

Holdout folds 3 and 4 add interrogative and reported-question constructions
absent from the new development material and from the four Alder development
suites. Their presence is fixed before capture and is not a post-hoc stress
test. The holdout file may be checksummed and passed by path to the dedicated
one-shot runner, but it may not be used for ordinary capture, diagnostics,
implementation decisions, or calibration.

## Phase A: zero-weight preflight and payload

Before a full model process:

1. reparse all three JSON suites and verify exactly one fact, the declared
   group counts, unique prompts, exact factorial cells, `T=4`, target IDs, and
   entity alias IDs;
2. verify the model architecture, final logit softcap, dimensions, vindex
   binding, source commit, binary hashes, suite hashes, and this document hash;
3. check the real process list and refuse to start if another E4B process is
   resident;
4. compile exactly one natural `COMPOSE` payload from the development fact.

The payload phase passes only if compilation yields one serializable source
slot bound to target ID `107545`. A compilation error or natural payload miss
is a scientific failure for this candidate. No alternate row, KNN mode,
prompt, target, or fact is tried.

## Phase B: development relation direction

Apply the source payload without a contextual detector and capture only L13 at
the last prompt position for all 24 development rows in one bounded process,
using `--residual-layer 13` without `--position-residuals`. Fit one direction
on all four factorial cells:

```text
center = mean(all development residuals)
mu_xy = mean(residual - center for cell xy)
direction = unit(0.5 * ((mu_11 + mu_01) - (mu_10 + mu_00)))
score(x) = dot(x - center, direction)
```

Before the all-development fit is accepted, run a fixed
leave-one-construction-out diagnostic. In each of the six folds, fit the same
formula on the other five groups and set training-only thresholds to the
relation-absent ceiling and relation-present floor. Every fold must have a
strict positive separation gap. On the six held-out `10` controls there must
be zero `fire` states; at least five of six held-out `11` positives must
`fire`; any remaining positive must `abstain`, not `no_fire`.

As a null diagnostic, rerun the complete grouped pipeline for all 64 possible
per-group relation-bit flips. Define the scalar primary score as positive fire
coverage when control fires are zero, and as minus the control-fire rate
otherwise. The observed unflipped score must be strictly above the 95th
percentile of that exact null distribution. Failure closes the experiment
before calibration.

The fixed grouped diagnostic command shape is:

```bash
python3 crates/larql-lql/examples/analyze_factorized_relation.py \
  development=/private/path/development-capture.json \
  --group-field fold \
  --layer 13 \
  --representation last \
  --method raw_conjunction \
  --threshold-policy separation_gap \
  --relation-flip-null
```

## Phase C: calibration and ternary thresholds

Freeze the development-fitted center and direction, then capture calibration
once. Calibration uses both entity values to set:

```text
no_fire_threshold = max(score for cells 10 and 00)
fire_threshold = min(score for cells 11 and 01)
gap = fire_threshold - no_fire_threshold
development_range = max(development scores) - min(development scores)
```

Acceptance requires `gap > 0`, `development_range > 0`, and
`gap / development_range >= 0.10`. The policy is then:

```text
no exact entity alias       -> compiled_reject
score >= fire_threshold     -> fire
score <= no_fire_threshold  -> no_fire
otherwise                   -> abstain
```

Only `fire` makes the payload and terminal adapter visible. All other states
retain the complete base path. Threshold equality is deliberately inclusive
at the two outer states and may not be moved after calibration.

If calibration passes, serialize the center, relation direction, thresholds,
two entity alias sequences, candidate metadata, and base binding. Reparse the
artifact and prove offline parity before changing the public `.vlp` ABI. An ABI
extension is authorized only to represent this already-frozen artifact; it may
not add another detector family or selectable representation.

The sole calibration compiler is
`crates/larql-lql/examples/compile_factorized_relation_v1.py`. It exposes paths
for the two captures and its report output, but no candidate hyperparameter.

## Phase D: fixed terminal adapter and development gate

With the detector-attached source patch frozen, every development and
calibration `11` row must be `fire`; every `10` row must be `no_fire` or
`abstain`; every `01` and `00` row must be `compiled_reject`. Any positive
which the earlier grouped diagnostic permitted to abstain must therefore be
resolved by independent calibration; it cannot be overridden.

Capture post-L41 last-position residuals for the fired positives. Run the
unchanged softcap-aware terminal compiler once: fixed gradient projection,
five-percent residual-norm cap, fixed 65-point monotonic grid, target margin
`+0.75`, and deterministic bisection. No alpha, layer, direction, scale, or
prompt sweep is allowed.

The development gate then requires:

- every positive chooses ID `107545` naturally with margin at least `+0.5`,
  emits exactly the four target IDs, arms once, and releases;
- every non-fired control applies neither payload nor adapter and retains
  complete f32 logits bit-identical to the corresponding base path;
- detector decisions and scores are bit-identical between reference and
  adapter paths;
- serialized artifacts reparse with the exact base binding and dimensions.

Any failure keeps the holdout sealed.

## Phase E: one holdout access

The dedicated holdout runner first verifies every other frozen artifact. It
then opens the holdout exactly once, verifies the bytes against the frozen
holdout checksum, and only then parses them in the same resident process.
Acceptance requires all of the following:

- zero `fire` decisions on all six `10` controls;
- all twelve `01` and `00` rows are `compiled_reject`;
- at least five of six supported `11` groups are `fire`;
- every uncovered `11` row is `abstain`, never `no_fire`;
- for fired positives, `(score - fire_threshold) / gap >= 0.10`;
- for `10` controls, `(fire_threshold - score) / gap >= 0.10`;
- every fired positive selects ID `107545` with margin at least `+0.5`, emits
  the exact four-ID span, arms once, and releases;
- every non-fired row retains the complete base path without payload or
  terminal-adapter application.

The fact-level verdict is reported before any aggregate prompt count. A single
failed condition closes factorized-relation-v1. There is no threshold move,
new alias, binary fallback, second adapter, prompt edit, extra holdout access,
or replacement fact.

## Resource and evidence contract

Every full E4B invocation is sequential and runs in a unique foreground
`systemd --user` unit with `--wait --pipe --collect`, after checking the host
process list. Fixed limits are:

- `MemoryHigh=22G`;
- `MemoryMax=24G`;
- `MemorySwapMax=8G`;
- `OOMPolicy=stop`.

Development and calibration capture units use `RuntimeMaxSec=1800`; terminal
compilation and development validation use `5400`; final holdout uses `3600`.
A cgroup kill, timeout, or runtime fault is infrastructure-inconclusive. It
does not authorize relaxed limits or a scientific retry.

Before phase A loads weights, a private provenance manifest records the exact
source commit; hashes of this document, suites, release binaries, tokenizer,
vindex manifest, and zero-weight preflight; the candidate configuration; and
the three unit names. Captures, detector, patches, journals, and raw holdout
evidence stay in the private research directory. This repository receives the
technical result and reproducible non-private protocol only.

## Stop rules

- No E4B run occurs until this preregistration, suite material, analyzer, tests,
  and preflight changes are committed and checksummed.
- Alder-v2 remains closed. Its 48 development captures may explain the
  architecture choice but may not fit or calibrate this patch; its holdout is
  never opened.
- Each declared E4B phase runs once. Any scientific gate failure stops every
  later phase.
- The outcome is `pass`, `scientific failure`, or
  `infrastructure-inconclusive`. `Almost`, binary accuracy, and a rescued
  candidate are not outcomes.
