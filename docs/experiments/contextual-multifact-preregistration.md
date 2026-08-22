# Contextual Multi-Fact Replication Preregistration

**Status:** completed without opening either final holdout. The frozen protocol
and candidate outcomes are reported in
[contextual-multifact-results.md](contextual-multifact-results.md).

This document was frozen before payload compilation, residual capture, or
native behaviour measurement for either candidate.

## Question

Does the model-bound detector → conditional FFN payload → `ActiveSpan`
protocol replicate beyond the single Galadriel association on the same exact
Gemma 4 E4B Q4K artifact?

This experiment is a bounded mechanism replication. It is not a scaling
study, an automatic-editing benchmark, or evidence of cross-model transfer.

## Fixed candidates

The candidates were selected before observing model outputs. They are not
replaceable after the experiment starts.

| Class | Entity | Relation | Target | Token IDs |
|---|---|---|---|---|
| Counterfactual | `France` | capital | `Rivendell` | `[44625, 643, 713]` |
| Synthetic association | `Project Alder` | custodian | `Ada Lovelace` | `[45158, 37937, 217262]` |

The counterfactual candidate tests behaviour conflicting with a familiar
fact. The synthetic candidate tests an association that cannot be expected to
come from pretraining. Both targets have the same three-ID length as the
existing Galadriel case. A passing result remains a conditional behaviour
edit; it is not described as knowledge acquisition.

## Frozen data split

Each candidate has three named development suites containing 12, 8, and 12
factorial prompts, followed by one 16-prompt final holdout. Each cell separates
entity and relation presence:

- `11`: intended entity and relation;
- `10`: entity only;
- `01`: relation only;
- `00`: neither factor.

The exact prompts live next to the public example harness:

- `multitoken_replication_france_{dev1,dev2,dev3,holdout}.json`
- `multitoken_replication_alder_{dev1,dev2,dev3,holdout}.json`

No final-holdout prompt may be used for payload compilation, detector
selection, threshold fitting, hard-negative mining, or implementation choices.
The holdout is opened once, after the patch is frozen.

## Fixed compilation protocol

1. Capture native first-target-ID rank and probability.
2. Run one `COMPOSE` payload compilation per candidate. Do not change the
   entity, relation, target, payload slot, or target tokenisation after seeing
   its output.
3. Capture all prompt positions for the three development suites.
4. Run the existing leave-one-entire-suite-out selector without changing its
   search space: 42 layers, ten deployable position representations, and four
   detector families. Acceptance requires every development decision correct
   and a positive worst signed margin.
5. Fit the accepted detector on all 32 development captures, bind it to the
   exact vindex checksum, and freeze the contextual patch.
6. If no detector is accepted, or the payload does not make the first target
   ID top-1, record that candidate as failed. Do not substitute another fact.
7. Open the final holdout and run the real cached Q4K path.

## Metrics and acceptance

Report candidate-level outcomes before prompt-level totals.

For each candidate:

- native and patched first-target-ID rank and probability;
- patched top-1 versus runner-up logit margin;
- weakest positive and closest negative detector margins;
- positive exact first ID, exact two-ID `ActiveSpan` continuation, and release;
- negative detector rejection, absent payload execution, absent first target
  ID, and unarmed span.

A candidate is certified only if all four positives and all twelve controls
pass the downstream cached contract. The report must say `certified facts /
attempted facts`; an aggregate such as `32/32` is secondary and must retain its
fact denominator.

If both new candidates certify individually, combine their operations in one
checksum-bound patch and rerun both frozen holdouts with both patches live.
This coexistence arm passes only if each positive selects its own target and
all other detectors reject, while every control remains rejected.

The historical Galadriel result remains a separately calibrated fact. A
successful replication can therefore support at most the statement “three
model-bound facts attempted and certified,” with per-fact breakdowns. It does
not support broad coverage or arbitrary edit compilation.

## Stop rule

The replication is limited to two working days. At expiry, publish the fixed
candidates and their actual outcomes, including compilation failures or an
opened-holdout failure. No extra candidate, threshold adjustment, or new
detector family is allowed inside this gate.
