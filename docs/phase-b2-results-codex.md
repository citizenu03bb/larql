# B1/B2 closeout: stop, no certified dev edit

All 15 primary edits and independent certifications completed under the
frozen method, with no failed services, retries, replacement facts or
scientific parameter changes. After Pat's separately recorded C2 coverage
arbitration, **L4, L8 and L12 each score 0/5**. The required 2/5 gate is not
met. No layer is selected, no identity-C ablation is run, and B3 remains
unopened. Claude reports **15/15 REFUSED**; an independent sufficient
mandatory-control witness supports each refusal.

## A qualification limitation prevents a clean method-level negative

Native qualification is identical across the three layers for every fact.
At the fixed 24-token cap, three facts lose their subject-after-relation
attribute question. Bolt loses all unseen efficacy questions in that
order. Only Eiffel has the native coverage needed to potentially pass
both C1 and the arbitrated C2:

| Public fact | Qualified unseen efficacy, after / before relation | Qualified attributes, after / before | Potential double pass |
|---|---|---|---|
| Beethoven | 2 / 2 | 0 / 1 | no |
| Bolt | 0 / 2 | 1 / 1 | no |
| Eiffel | 2 / 2 | 1 / 1 | yes |
| Kilimanjaro | 2 / 2 | 0 / 1 | no |
| Toyota | 2 / 2 | 0 / 1 | no |

The resulting **native-coverage upper bound is 1/5**, already below 2/5,
even for hypothetical perfect edited answers. This is a post-comparison
feasibility diagnostic, not a pre-registered criterion or a rescoring of
the archived run. The protocol should have qualified all native questions
and checked gate feasibility before solving edits. The mandatory stop is
correct; describing this as a clean demonstration that ROME cannot meet
the target would be too strong. Actual failures on qualified questions
remain failures. There is no certified dev fact in this run.

The complete [native feasibility receipt](research-data/phase-b2-native-feasibility-codex.json)
and [native qualification receipt](research-data/phase-b2-native-qualification-codex.json)
preserve the calculation. A follow-up proposal is separate from the run;
see [evaluation follow-up](phase-b2-evaluation-followup-proposal.md).

## The C2 amendment materially changes the gate

| Layer | Original reported C1+C2 PASS | Decision after Pat's C2 arbitration |
|---|---|---|
| L4 | 0/5 | 0/5 |
| L8 | 2/5 | 0/5 |
| L12 | 1/5 | 0/5 |

The original reported L8 score would clear the development gate. Its two
cases are Beethoven and Toyota, both missing one native-qualified
attribute order. Pat chose to require both orders after the first
Beethoven discrepancies were presented, before the complete results.
This post-comparison amendment is explicit and changes the decision; it
is not presented as the original frozen rule. Original harness verdicts,
checker code/output and protocol remain intact.

Across 30 independent C1/C2 comparisons, 25 statuses agree exactly. The
five differences are all C2 PASS versus NOT_EVALUATED for missing native
order coverage: Beethoven L8/L12, Kilimanjaro L4, Toyota L4/L8. They are
covered by Pat's arbitration. No other control-reading difference in the
checked C1/C2 scope remains unresolved. The full original and amended
statuses are retained in the [comparison](research-data/phase-b2-gate-comparison-codex.json)
and [decision receipt](research-data/phase-b2-decision-codex.json).

## Independent checks and their limits

Claude independently recomputed all 15 saved algebra constructions and
reported them correct. Each certificate's plan points to the original
Codex export; its backend and U0/V0 files before and after reload are
byte-identical to that export. All 510 archived generations decode back
to their saved texts with the pinned local tokenizer. EOS, cap flags,
record/evidence/verdict hashes, NF4/source fingerprints and actual
forward counts match. The 5,765 generated tokens plus 120 diagnostics
recount to **5,885 certification forwards**.

The C1/C2 checker was frozen at `1d6af9e8` before reading expected peer
statuses. Twelve logical edge-case tests passed across the gate checker,
separate arbitration, and refusal-witness extension. The latter extension
was written after partial results were visible and is labelled as such.
It proves each REFUSED verdict with a sufficient mandatory-control
non-PASS; it does not exhaustively replay all C0--C8 statuses or validate
positive certifications. The peer's complete C0--C8 counts in the summary
are reported results, not independently reconstructed statuses.

Claude reports C0 and C7 PASS for all 15 adapters, C1 PASS for only three,
C3 non-PASS for all 15, and C6 FAIL for eleven. Stored adapter form and
reload consistency do not establish a selective fact edit. Peer service
fingerprints omit a full checkpoint hash; their metadata/source hashes
match B1, and the unchanged live checkpoint stat matches B1's separate
full-hash preflight. This limitation is preserved in the audit.

## Cost and delivery

| Work | Real forwards | Adam updates | Model services |
|---|---:|---:|---:|
| B1 covariance | 200 | 0 | 1 |
| 15 B2 solves | 1,950 | 375 | 15 |
| Independent primary certifications | 5,885 | 0 | 60 |
| Total | **8,035** | **375** | **76** |

The registered upper bound was 18,790 forwards and 96 services including
the conditional identity ablation. Model-free checking added no model
passes. Every real service used the shared flock and bounded systemd
launch. At closeout, no heavy model process remained on the GPU; only
the pre-existing lightweight router was present. No push was made.

The [compact summary](research-data/phase-b2-summary-codex.json),
[record audit](research-data/phase-b2-service-records-audit-complete-codex.json),
[refusal witnesses](research-data/phase-b2-refusal-witnesses-complete-codex.json)
and [observed peer index](research-data/phase-b2-peer-index-observed-codex.json)
pin the input hashes. Full comparisons and raw method artifacts remain
in `~/Projects/larql-research/phase-b-rome-20261026/`; raw independent
certification records remain in
`~/Projects/larql-research/phase-b-certification-20261026/`.
