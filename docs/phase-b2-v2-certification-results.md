# Phase B2 v2: re-certification of the 15 development edits, amended readings

*Claude Code, 2026-10-26. Protocol: `docs/phase-b2-v2-recertification-protocol.md`, approved by Pat; amendment A1, a driver repair, at `07c8d94d`. These are the same 15 ROME rank-1 edits as in v1 (E2B NF4, blocks 4, 8 and 12), with no new solve. Development set only. The blinded test set was not opened.*

## Summary

- **The feasibility gate passes: 4/5 facts.** At 64 tokens, Beethoven, Bolt, Eiffel and Toyota qualify natively in both subject orders, for efficacy and for attributes. Kilimanjaro does not, because its after-relation attribute is still truncated natively.
- **All 15 edits are `REFUSED`.** 7 593 forwards in total, no service failure.
  - Every verdict was recomputed model-free from the raw records: 15/15 identical.
  - Amendment A1 was needed. In run 1, the driver had skipped the edited and reload services of the five L4 cases. They were resumed with their native services reused; nothing else changed.
- **Provisional B2 gate:** the number of facts with C1 PASS and C2 PASS is **0/5 at L4, 0/5 at L8 and 0/5 at L12**. This is below the 2/5 threshold. Under Codex's protocol, B2 stops and ROME rank 1 is not selected; the stop is Codex's to confirm. Claude computed the score mechanically; Codex cross-checks it model-free.

## Verdicts

| Edit | C1 | C2 | C3 | C6 | Decisive reasons |
|---|---|---|---|---|---|
| Beethoven L4 | FAIL | FAIL | FAIL | PASS | no flip; attribute lost |
| Beethoven L8 | **PASS** | NOT_EVAL | NOT_EVAL | FAIL | bleed: Mozart, Chopin → Italian |
| Beethoven L12 | **PASS** | NOT_EVAL | FAIL | FAIL | bleed: Mozart, Bach, Chopin → Italian; "Moonlight Sonata … by Giuseppe Verdi" |
| Bolt L4 | FAIL | PASS | FAIL | PASS | no flip |
| Bolt L8 | FAIL | FAIL | FAIL | PASS | no flip; attributes lost |
| Bolt L12 | **PASS** | FAIL | FAIL | FAIL | Jamaica → "switzerland", nickname → "swimmer"; Federer → swimmer |
| Eiffel L4 / L8 / L12 | FAIL | PASS / FAIL / FAIL | FAIL | PASS | no flip at any layer |
| Kilimanjaro L4 / L8 / L12 | FAIL | NOT_EVAL / FAIL / FAIL | FAIL | FAIL | no flip; bleed |
| Toyota L4 | FAIL | FAIL | FAIL | PASS | no flip |
| Toyota L8 | **PASS** | NOT_EVAL | FAIL | FAIL | bleed: Honda → Germany; "Volkswagen produces the Corolla" |
| Toyota L12 | FAIL | FAIL | FAIL | FAIL | no flip; bleed |

C0, C4 and C7 pass on every case. The full statuses, forward counts and hashes are in `docs/research-data/phase-b2-v2-certification-index-claude.json`, a byte copy of the run index.

## What v2 settles, compared with v1

- **The efficacy is real and robust to rephrasing** at L8 and L12, for nationality and country of origin. Every unseen phrasing, in both subject orders, states the new value from an empty answer.
- **The bleed is real, not an artifact of the wording.** C6 now compares declared values, and it still fails: neighbours of the same type take the edited value.
- **The entity substitutions are real and now caught** by the subject-naming rule on answer-only mentions: "Giuseppe Verdi", "Volkswagen".
- **Where C1 passes, the edit also damages the free answers about the subject's other attributes.**
  - The edited answer to the after-relation attribute becomes long and hedging ("There is no single person who founded Toyota…") and runs past 64 tokens. C2 is therefore NOT_EVALUATED, not PASS.
  - On Bolt L12, the attributes are simply corrupted.

## Reading, as fixed in the protocol

The gate is below 2/5. **ROME rank 1 at blocks 4, 8 or 12 does not reach, on this dev set and instrument, paraphrase-robust efficacy together with intact attributes.** Where it reaches efficacy, it is not selective: neighbours bleed, entities are substituted, and attribute answers degrade. This is a documented negative about ROME rank 1 on Gemma 4 E2B NF4, with the evidence to support it. It is not a claim about other methods or models.

## Integrity

- One heavy model at a time, under the shared lock. Limits per service: 22G/24G/8G and 1800 s, with at most 400 forwards (Q1). No retry and no parameter change.
- The single implementation repair (A1) is recorded in the protocol and in the index (`repairs`).
- Raw records, evidence, verdicts, plans and the driver log are in `~/Projects/larql-research/phase-b-certification-v2-20261026/`. They are archived as `phase-b2-v2-certification-20261026.zip`, with a SHA-256 manifest and a restore check.
- Verification so far is the same-agent recomputation. Codex's model-free cross-check of the evidence and of the gate score is pending.
