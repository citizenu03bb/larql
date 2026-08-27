# Progressive evidence protocol for contextual multi-token editing

**Status:** active protocol for future research arcs

**Scope:** model-bound contextual editing research; prior frozen verdicts remain
unchanged

## Purpose

The earlier protocol correctly protected certification but applied some of its
one-shot rules during mechanism discovery. This document separates those jobs.
Rigidity increases with evidential value:

```text
exploration -> nomination -> shakeout -> prospective replication -> certification
```

An exploratory result may generate a candidate. It may never certify the
material which generated it. A failed certified candidate remains failed even
if later exploration explains the failure.

## Evidence labels

Every artifact and reported number carries exactly one label:

- `exploratory`: generated from spent or inspected material;
- `nominated`: a closed candidate specification, not evidence;
- `shakeout`: pipeline and coverage evidence on disposable material;
- `prospective`: first untouched evaluation of the frozen candidate;
- `certified`: one-shot result on the sealed final material;
- `product-bounded`: compiled, enumerated behaviour without a model-bound
  semantic-generalization claim.

The label cannot be upgraded retrospectively. Recomputing more activations for
the same construction groups does not create new observations.

## Phase 0: exploration

### Material

Spent development captures, failed candidates, public data, and explicitly
disposable diagnostic material may be reused. Holdouts and calibration sets
from closed experiments remain inaccessible.

The effective sample size is the number of independent construction groups,
facts, or other declared sampling units, never the prompt-row count obtained by
crossing factorial controls.

### Freedom

Representations, references, layers, transformations, normalizations, and
diagnostics may be explored. Every executed configuration and every inspected
metric is logged. There is no p-value, certification claim, or hidden
best-of-k result in this phase.

Exploration should still prefer one mechanism-motivated experiment at a time.
A sweep is allowed only as a labelled map of the spent material, not as a
candidate victory.

### Permitted conclusions

- a mechanism is plausible or contradicted on spent material;
- a specific estimator failed on that material;
- one closed candidate is worth nominating prospectively;
- no candidate is justified.

Forbidden conclusions include generalization, certified precision, a deployable
threshold, and causal identification from observational correlation alone.

## Phase 1: nomination

Nomination consumes no new data. It produces a checksum-bound specification
containing:

- complete score function and all preprocessing;
- model artifact, layer, positions, aliases, dimensions, and ABI;
- sampling unit and material generator;
- primary metric, worst-case margin, and equality/NaN semantics;
- configuration count inherited from exploration;
- falsifier and stop rule;
- allowed abstention and fail-closed behaviour;
- minimum prospective sample size and exact claim boundary.

The candidate may not contain `best of`, an open-ended reference list, or a
hyperparameter chosen after prospective material exists. Nomination is a debt,
not evidence.

## Phase 2: shakeout

Shakeout uses fresh but permanently non-evidential material. Its job is to find
implementation defects before evidence is consumed:

- every declared alias, entity, relation surface, construction, and factorial
  cell is exercised;
- serialization, dimensions, dtypes, deterministic replay, and error paths are
  checked;
- no scientific score is promoted from shakeout.

Implementation bugs may be corrected freely. A change to the estimator,
metric, representation, reference, threshold policy, or claim returns the work
to nomination under a new protocol identifier. Shakeout material remains
burned.

## Phase 3: prospective replication

Prospective material is created only after nomination and shakeout finish. The
candidate, data generator, grouping, primary metric, and rule of decision are
frozen before the first value is read.

No candidate correction is allowed. An infrastructure fault may be retried
only when the fault is independent of model output, the bytes remain sealed,
and the retry policy was predeclared. A scientific miss closes the nominated
candidate on this material.

The report leads with per-group worst-case margins and safety violations. Means,
AUC, and aggregate prompt accuracy are secondary diagnostics only.

## Phase 4: calibration and certification

Calibration and certification use disjoint, untouched material. Calibration
may set only the already-declared deployable parameters, once. The final
holdout is opened once by a checksum-bound runner after every other artifact is
frozen.

Certification is an absolute gate for one candidate, never selection among
candidates. Equality, NaN, an undeclared alias, and an unsafe activation fail
closed. An abstention is safe but counts against coverage exactly as declared.

The final verdict is `certified`, `scientific failure`, or
`infrastructure-inconclusive`. `Almost` and rescued thresholds are not verdicts.

## Corrections and protocol defects

Corrections are classified before another statistic is inspected:

1. **Mechanical completion:** a predeclared arm could not execute because of an
   independently observable implementation defect, such as an incomplete alias
   table. It may be completed once under a new protocol identifier, with the
   original estimator and rule unchanged.
2. **Estimator change:** a new representation, learned direction, reference,
   normalization, metric, or threshold created after seeing an outcome. It is a
   new exploratory degree of freedom and cannot complete the old protocol.
3. **Infrastructure fault:** process, memory, filesystem, or deterministic
   replay failure independent of the scientific values. It has no scientific
   sign but follows the frozen retry policy.

If values were already inspected, any correction runs as exploration. If no
target statistic was produced and the defect is mechanical, the unchanged
material may complete the missing arm once, but the result still inherits the
selection history of its lineage.

## Detector and effecter

Detector/effecter coupling is a named research hypothesis, never an implicit
safety mechanism. Observed co-failure may reflect a shared cause, a construction
nuisance, or chance. A product path must use an explicit gate: abstention means
no edit regardless of whether the effecter would also have failed.

Coupling diagnostics are non-gating secondary analyses on prospective material
or separately preregistered causal interventions. They cannot reinterpret a
failed detector gate.

## Product branch

The compiled product branch may progress independently with exact entity
aliases, enumerated relation surfaces, a bounded terminal effect, and
`ActiveSpan`. It receives the `product-bounded` label. Product success is not
evidence for semantic relation detection, and product evaluation material is
not reused for prospective model-bound claims.

## Permanent safeguards

- Alder-v2 and factorized-relation-v1 remain closed under their original rules.
- Their holdouts remain sealed.
- Configuration counts remain visible.
- No failed gate is relaxed because the miss is small.
- No exploration result is promoted without fresh grouped material.
- `No architecture justified` remains a valid and expected outcome.
