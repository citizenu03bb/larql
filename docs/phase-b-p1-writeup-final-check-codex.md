# P1 revised write-up: final check — Codex, 2026-10-27

Reviewed `docs/phase-b-p1-writeup.md` at
**da20f0739fca7c87ee04a5194a704e45b4cc39ca**, against the twelve points of
30d0d7ea and the unchanged measured receipts. Draft SHA-256:
`6a713d7b9633ccd5ea9718dfdd4991920ec3dbef87fab461f0cb5cf6d4bdf242`.
Line numbers below refer to the revised draft. No draft edit, model pass,
publication or push was performed.

**Eleven points are applied; one is partly applied.** No new numerical
discrepancy was found. One chronology correction and two short scope
clarifications remain below.

| Original point | Status | Revised evidence / remaining issue |
|---|---|---|
| 1. Seven mandatory controls out of nine | Applied | Lines 7, 73–75 correctly distinguish mandatory and diagnostic controls. |
| 2. Native W1 gap and 3/5 ceiling | Applied | Lines 21, 42, 65–71 and 142 disclose both ineligibility and ceiling; C2 is no longer the only stated obstacle. |
| 3. Qualified phrasing scope and spurious qualification | Applied | Lines 40, 95–96 explicitly scope the rule, Gretzky's excluded E1 and ES/PS denominators. |
| 4. Probability proxy versus published CounterFact evaluation | Applied | Lines 15, 61, 115–128 and 143 disclose adaptations, procedural differences and historical-only comparison. |
| 5. Nintendo / measured outputs versus causal mechanism | Applied | Lines 101, 105 distinguish the two factor constructions from Codex-only generation, and avoid the position-zero causal claim. |
| 6. Convention-declaration chronology | **Partly applied** | Lines 164–167 are correct, but line 59 still puts declaration of each agent's precision conventions before “Claude then rewrote”. |
| 7. Random counterpart is not inert | Applied | Line 97 states no flips, a nonzero margin change and the tested-generation scope. |
| 8. Scope of independent verification | Applied | Lines 153–156 distinguish independent implementation, known headlines, arithmetic replay and primary-source checking. |
| 9. Include NS >= 75% in the fixed criterion | Applied | Line 128 lists all three thresholds. |
| 10. Double KL weighting concerned loss, not budget | Applied | Lines 170–171 separate the two errors. |
| 11. Mixed outcome is outside the original reading table | Applied | Lines 25 and 129–133 attribute closure to Pat rather than inventing a pre-registered mixed row. |
| 12. Mark shortened answers as excerpts | Applied | Line 79 and the table do so explicitly. |

## Remaining corrections

**Line 59, chronology.** The sequence still reads as if all detailed
precision conventions were declared before Claude rewrote and ran A3.
The accurate account later in the draft does not remove this local
contradiction. Suggested replacement:

> We pinned the reference and retained A1's fixed 20-update policy. Claude
> rewrote its solve. The detailed numerical conventions were documented
> later, at the participant-specific times described below, before the
> new comparison. The two solves agreed on all five facts.

**Line 139, new causal/general wording.** “A pinned reference makes the
implementations converge” attributes the outcome to pinning alone and
reads as a general guarantee. The observed sequence included correcting
the algorithm and rewriting the implementation. Also, 0.03% refers to
update norms, not all numerical behaviour. Suggested replacement:

> After aligning the implementations to the pinned equations and declared
> adaptations, these five cases agree: minimum factor cosine 0.99999 and
> maximum update-norm difference 0.03%.

This is a claim-scope correction, not another experiment or a change to
the agreement gate. The revised write-up's quantified interpretation and
description of Codex's work are otherwise faithful to the checked data.

**Line 111, clarify the unknown case.** “It is not a collateral-change
case” can be read as an absence of collateral change. Here C2 is unknown.
The intended classification is clearer as **“We do not retain it as a
demonstrated collateral-change witness.”** The preceding sentence already
states the unknown status, so this is a local clarification rather than
a new disagreement about the measurement.

## Companion documents

The `phase-b-p1-5-results.md` and `phase-b-p1-closure.md` diffs correctly
carry the requested CounterFact qualification and W1/C3 caveat, with a
labelled correction. This is not a certification that every historical
sentence in those two documents has also received all twelve revisions.

The 0/5 verdicts, 5/5 A3 agreement, 45 control statuses, 111 margin sums,
pooled scores and 9938/13364 budget figures remain consistent with the
previous independent audit. No implementation changed; no new tests were
needed or run for this documentation-only check.
