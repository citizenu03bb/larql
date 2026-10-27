# Phase B2: closure, and the two follow-ups decided

*Claude Code, 2026-10-27, on Pat's decision of the same day ("clôture de B2, diagnostic et contrôle positif"). No model forward was run for this note.*

**Mandate line.** B2 is closed as a documented negative. The two follow-ups ask what the negative is a negative of: the edits (D1, a diagnostic with no new solve) and the bench and implementation (P1, a positive control where ROME is known to work). Each is a new decision, with its own protocol and approval. Neither resumes B2.

## Decision

- **B2 stops on its pre-registered rule.** No layer reaches 2/5 facts with C1 PASS and C2 PASS. ROME rank 1 at blocks 4, 8 and 12 is not selected. B3 (method freeze) and B4 (test set) do not run. The 20 test facts stay sealed.
- **The stop stands whatever Codex's pending model-free cross-check of v2 finds about individual statuses.** A discrepancy would be reported next to this note, not used to reopen B2.

## What is reported, as a vector per control rather than one score

Statuses of the 15 v2 edits (5 dev facts), from the run verdicts:

| Control | L4 | L8 | L12 |
|---|---|---|---|
| C1 efficacy, unseen phrasings, both orders | 0 PASS / 5 FAIL | 2 PASS / 3 FAIL | 2 PASS / 3 FAIL |
| C2 other attributes, both orders | 2 PASS / 2 FAIL / 1 NOT_EVAL | 0 PASS / 3 FAIL / 2 NOT_EVAL | 0 PASS / 4 FAIL / 1 NOT_EVAL |
| C3 mentions before and after the relation | 5 FAIL | 4 FAIL / 1 NOT_EVAL | 5 FAIL |
| C6 off-target | 4 PASS / 1 FAIL | 2 PASS / 3 FAIL | 1 PASS / 4 FAIL |
| C0, C4, C7 | 5 PASS each | 5 PASS each | 5 PASS each |

Readings that the write-up must keep:
- **The gate score is fragile.** At L8, Beethoven and Toyota have C1 PASS with C2 NOT_EVALUATED: their edited attribute answers run past 64 tokens. Had both been C2 PASS, L8 would have scored 2/5. NOT_EVALUATED does not count as PASS under the gate rule, so the stop is correct. It still rests partly on non-evaluations.
- **The robust finding is about the four edits that flip.** Each of them fails C6, with neighbours of the same type taking the new value, and C3, through entity substitution in the answer-only mention.
- **Efficacy is the majority failure.** C1 is not PASS on 11 of the 15 edits, including Eiffel at every layer and every L4 edit. C3 fails on those edits as well, because it requires the new value in both subject orders.
- **The bench has never certified a real edit.** The five archived replays and the 15 B2 cases are all REFUSED; only the fake-instrument tests certify. That is the limitation P1 addresses.
- **The check between the two agents is internal.** The controls, neighbour lists and caps were written by the same agents. Codex's cross-check is a black-box replay, not a third-party validation.
- **The v2 amendments are post hoc** relative to the original pre-registration. They were written after the v1 data and before any v2 forward, and applied to every case alike.

## Quorum consulted before this decision

- **Composition.** Three blind runs of the same question, ten answers from nine models: gemini-3.8-flash, gemini-4-argon, deepseek-flash, mistral-large-4 (two samples, the second served by Le Chonk), glm, kimi, qwen, grok, minimax. Codex and Opus were excluded as authors of the judged work.
- **Framing.** The framer had already recommended closure before writing the question. This is disclosed in the audit file.
- **Two framing defects were found after launch.** C3 was described too narrowly, and the model-free algebra cross-check was omitted. Neither changed a recommendation.
- **Convergences.**
  - All ten answers agree: close B2, keep the test set sealed, no new method now.
  - Eight point to the key geometry of subject against neighbours as the cheapest discriminating measurement.
  - Six reject the neighbour-constrained variant (B2b), each for its own reason.
  - Six ask for a positive control before the negative can be interpreted.
- **Where the record is.**
  - The synthesis and audit are copied in `docs/research-data/phase-b-next-step-quorum-synthesis-20261027.md` and `docs/research-data/phase-b-next-step-quorum-audit-20261027.md`.
  - The raw answers are archived as `phase-b-next-step-quorum-20261027.zip` (35 files, SHA-256 manifest, restore check; archive SHA-256 `caf27aed…b323`).

## Follow-ups

- **D1: diagnostic of the existing 15 edits.** Protocol `docs/phase-b-d1-diagnostic-protocol.md`. Capture and teacher-forced forwards only; no generation and no new v\* solve. It separates three causes:
  - the edit does not fire at the subject in unseen phrasings;
  - it fires, but its value does not move the answer;
  - it moves the answer but is silent under greedy generation.

  D1 also measures whether the dose that reaches neighbours predicts the observed bleed, and how that dose changes with three covariances: the B1 one, the identity, and one estimated on a small Wikipedia sample (Pat's suggestion, which needs a download approval).
- **P1: positive control.** Design brief `docs/phase-b-p1-positive-control-design.md`. It runs the bench, and the solver, on a model and facts where ROME's published behaviour is known, so that the bench's ability to certify and the solver's fidelity are tested apart from E2B NF4.
