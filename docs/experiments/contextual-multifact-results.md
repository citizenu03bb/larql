# Contextual Multi-Fact Replication Results

**Date:** 2026-08-22  
**Protocol:** [frozen preregistration](contextual-multifact-preregistration.md)  
**Outcome:** 0/2 new candidates certified; both final holdouts remained sealed

## Result at a glance

The bounded replication did not reproduce the complete contextual multi-token
protocol on either pre-registered association. This is a candidate-level
result, not a prompt-level pass-rate claim.

| Candidate | Payload development result | Detector result | Final holdout |
|---|---|---|---|
| `France` / capital -> `Rivendell` | failed: target first ID top-1 on 1/3 dev1 positives | not fitted after payload failure | sealed |
| `Project Alder` / custodian -> `Ada Lovelace` | failed: target first ID top-1 on 7/8 development positives | failed: best fixed selector 31/32 | sealed |

Including the separately calibrated historical Galadriel result, the bounded
program therefore has **1 certified fact / 3 attempted facts**. The existing
Galadriel holdout remains 4/4 intended prompts plus 12/12 factorial controls;
it is not reinterpreted as evidence of arbitrary edit compilation.

## Counterfactual candidate: France -> Rivendell

The native prompt `The capital of France is` strongly preferred ` Paris`
with probability `0.997192`; the target first ID ` Riv` was absent from the
captured ranking. The native top-1 margin was `+6.2581` logits.

The single permitted `COMPOSE` compilation installed one payload at
L31/F1625. On the three dev1 positives:

- `The capital of France is`: ` Riv` rank 2, probability `0.134312`,
  `1.84777` logits behind ` Paris`;
- `France's capital is`: ` Riv` rank 1, probability `0.504565`, margin
  `+0.170408`;
- `The capital city of France is`: ` Riv` rank 2, probability `0.214644`,
  `1.27589` logits behind ` Paris`.

Because the payload did not reliably make the first target ID top-1, the
candidate failed before detector fitting. No threshold was adjusted, no
candidate was substituted, and the final holdout was not opened.

## Synthetic candidate: Project Alder -> Ada Lovelace

The native canonical prompt returned ` a` with probability `0.318523`; the
target first ID ` Ada` was absent from the captured ranking. Its independently
compiled payload also selected L31/F1625.

The payload made ` Ada` top-1 on 7/8 development positives. The sole failure
was the passive formulation `Project Alder is kept by`: ` the` remained top-1
at `0.324535`, while ` Ada` was rank 2 at `0.303575`, a deficit of only
`0.066763` logits.

The pre-registered leave-one-entire-suite-out selector searched the unchanged
42-layer, ten-representation, four-family space. No candidate met its 32/32
acceptance rule. The best candidate used L6 `mean_no_bos` residuals with unit
factor prototypes and achieved 31/32, AUC `0.994792`, and worst signed margin
`-0.062975`. It missed the same passive positive as the payload.

The detector signal is strong but the certification bar was not met. Its sole
error co-localised with the payload deficit: this surface form was neither
addressed reliably by the held-out early residual representation nor written
strongly enough by the later FFN row.

As a post-hoc diagnostic, fitting that selected detector on all 32 development
captures classified the same 32 captures with minimum signed margin
`+0.116526`. This is an in-sample deployment fit, not a generalisation
estimate. Under the frozen protocol it would have been the normal fitting step
only after leave-one-suite-out acceptance; it cannot replace the failed 31/32
gate. Freezing it as a rescue or moving its threshold would violate the
protocol. The final holdout therefore remained sealed.

## Additional observations

Both independent compilations selected the same L31/F1625 payload slot. The
pre-registered coexistence arm required two individually certified candidates,
so it was not run. The collision nevertheless identifies a separate scaling
constraint: independent single-edit compilation does not currently guarantee
composable, non-colliding storage.

The failures support three narrower hypotheses for future, separately
pre-registered work:

1. A one-row payload has a fact-dependent capacity limit governed partly by
   the native target logit deficit.
2. The chosen FFN slot and linear address are sensitive to syntactic surface
   form, especially changes in the final-token construction.
3. Payload compilation, contextual addressing, and multi-edit allocation need
   joint acceptance contracts rather than independent best-effort stages.

## Recommended next experiment

Do not repair either frozen candidate inside this experiment. A subsequent
`Alder-v2` must be a separately pre-registered experiment, not a correction of
this result. Its engineering phase should develop a stronger payload mechanism
away from the sealed Alder holdout. Before compiling Alder-v2, freeze a fourth
development suite stressing passive and agentless constructions, a numeric
first-token margin requirement, the unchanged all-suite leave-one-suite-out
detector bar, and an exact stop rule. Commit those artifacts and record their
commit hash before the first new capture.

A broader compiler study should stratify candidates by native target deficit
and return an explicit `UncompilableEdit` result when the available payload
budget cannot meet its margin contract. Multi-row payloads or a small
conditional residual adapter are the most direct capacity extensions; forcing
the first token after `ALLOW` is a distinct capability and must give up the
current "model-selected first ID" claim.
