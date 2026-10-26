# R4 C0: Codex's reading after comparison

This is Codex's methodological recommendation, requested by Pat after the
v2 comparison (`18bef9ec`). It records the historical discrepancy and a
preferred interpretation. It does not record a final decision by Pat or
replace the independent freeze (`5b85fba8`), previous comparisons, or their
three reported R4 form mismatches.

**I recommend retaining the harness's three values.** C0 describes the
candidate's application rule. R4's candidate is full prompt-subject
substitution, hydrogen to helium. The native answer prefix belongs to the
measurement procedure used to observe that candidate.

| Property | Frozen Codex | Harness | My preferred reading and reason |
|---|---|---|---|
| Response-time trigger | true | false | **false**: subject substitution is an input transformation; it does not wait for an answer state to trigger. |
| Uses native proposal | true | false | **false**: the substitution rule needs no native proposal to choose its timing or action. The archived measurement uses a native prefix, which remains a limitation of its evidence. |
| Applied at every forward | false | true | **true for the declared candidate rule**: the substituted prompt is retained as the input context, with no answer-time pulse or release. This does not establish an observed multi-step or reloaded execution. |

My frozen R4 interpretation classified the measurement procedure instead
of the candidate rule. That was inconsistent with my treatment of R2:
I allowed R2's static rule to pass C0 even though its measurements used
resumed native boundaries. The same distinction should apply to R4. The
existence of a native measurement prefix does not itself make a candidate
rule depend on a native proposal.

The every-forward value has a precise scope: it describes the declared
subject-substitution rule, not a persistence result inferred from one
saved frontier. If the candidate were instead explicitly defined as a
one-time donor intervention followed by restoration of the native prompt,
it would be a different rule and would need its own form description.
That is not the R4 candidate defined here. C0 must not be used to promote
this distinction into evidence of free generation or reload persistence.

This changes my interpretation of three metadata fields, not the measured
numbers or any control status. The uncontested R4 properties still give
C0 **FAIL**: donor pass, subject substitution and no unique stored artifact.
C2 and C3 remain **FAIL** on the number frontier, C7 remains
NOT_EVALUATED, and the verdict remains **REFUSED**. No persistent edit or
edit of a fact is established.

The historical disagreement stays visible in the
[v2 result](research-data/edit-cert-replay-blackbox-comparison-A1-A2-v2-codex.json)
and [comparison note](edit-cert-replay-comparison-v2-codex.md). This separate
note supplies my reading for Pat's arbitration. No comparator, harness,
freeze, previous report or archived result was changed. No model or
harness rerun was needed.
