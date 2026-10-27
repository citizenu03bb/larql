# Phase B P1-1 v2: native qualification on GPT-2 XL with rewritten templates — results

*Claude Code, 2026-10-27. Protocol `docs/phase-b-p1-v2-protocol.md`, approved by Pat. Native generations only; no edit has been solved or applied in P1.*

## Result: the gate opens, exactly at its threshold

- **5 of 10 facts are feasible:** BMW, Germany, Gretzky, Nintendo and Shakespeare. The threshold was 5, so P1 goes on to P1-2 and P1-3, after Pat's approval of their amendment.
- **The run:** 10 services and 478 forwards, with no failure and no cap hit.

| Fact | P1-1 (v1) | P1-1 v2 | What decided it in v2 |
|---|---|---|---|
| BMW | feasible | feasible | E1, E3, E4 state Germany; E2 now rambles ("now a member of the European Union"), and the other three still cover both orders |
| Nintendo | feasible | feasible | all four efficacy phrasings and both attributes qualify |
| Germany | no | **feasible** | the rewritten A1 gives "Berlin." |
| Gretzky | no | **feasible** | the rewritten A2 gives "Canada."; E2, E3, E4 state hockey |
| Shakespeare | no | **feasible** | E1 "England.", E4 "England," |
| Times Square | no | no | the rewritten A1 still rambles ("a great place to see the city's nightlife") |
| Great Wall | no | no | the rewritten A1 is wrong ("Han Dynasty"; Ming was declared) |
| Everest | no | no | the after-order efficacy prompts still ramble, and so does the rewritten A1 ("world's highest peak.") |
| Leonardo | no | no | only E2 states Italy |
| Moscow | no | no | no efficacy phrasing states Russia. "not Russia." is now rejected by the negation rule (§15), as intended |

## Two points to read with the result

- **The gate opens at the threshold exactly. One feasible fact rests on part of a spurious answer.**
  - Shakespeare's E2, "*not known to the British government until the late 19th century.*", qualifies because "British" is a declared value and the negation lies three words away, one beyond the two-word rule.
  - Without E2, Shakespeare is still feasible: E1 and E4 state England and cover both orders.
  - E2 will, however, be a required C1 question for that fact in P1-5. The bench will then demand the new value in an answer that never stated the old one. That makes C1 stricter, not looser, so it cannot produce a false certification. It will be reported with the verdict.
- **Prediction against outcome.**
  - Predicted 5–7 feasible; observed 5.
  - BMW, Nintendo and Germany were predicted likely, and are feasible. Times Square was predicted likely, and is not.
  - Shakespeare was predicted unlikely, and is feasible. Leonardo was predicted unlikely, and is not.
  - That one of BMW and Nintendo might be lost through the generic rewrite did not happen: BMW lost E2 but kept its feasibility.

## Next (needs Pat's approval of an amendment before any forward)

- P1-2, the covariance at block 17 (Codex's code, run by Codex, on WikiText-103).
- P1-3, the shared context templates, then the two independent solves of the five feasible facts.

Their details, budgets and the per-service count for a batched solve are set in the amendment `docs/phase-b-p1-amendment-a1.md`.

## Files

- Outputs: `~/Projects/larql-research/phase-b-p1-v2-20261027/qualification/`.
- Index copy: `docs/research-data/phase-b-p1-v2-qualification-index-claude.json`.
- Archive: `phase-b-p1-v2-qualification-20261027.zip`, with a SHA-256 manifest and a restore check.
