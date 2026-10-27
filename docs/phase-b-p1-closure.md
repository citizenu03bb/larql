# Phase B P1: closure as a mixed result

*Claude Code, 2026-10-27, on Pat's decision of the same day: "Je clos P1 en résultat mixte : critère global non atteint, cause de B2 non établie, BMW et Gretzky comme témoins de l'effet sur les attributs de l'entité." No model forward was run for this note.*

**Mandate line.** P1 was the positive control of the certification bench. It asked whether the bench can certify a real edit made by a method that is known to work. It closes without a certified edit. It does leave two reproduced witnesses of what the bench sees and the literature's metrics do not.

## Decision

- **P1 closes as a mixed result.**
  - The global positive-control criterion is not met.
  - The cause of the B2 negative is not established.
  - BMW and Gretzky are kept as witnesses of the effect on the entity's other attributes.
- **No further tuning, solve or model pass is run under P1.** This includes rewording questions, raising the 24-token cap and changing facts. The ten candidates and the two P1-4 gates (0/5 under A1, 5/5 under A3) are kept as they are.

## What P1 established, with its measurements

1. **The construction of the edits is reproduced.**
   - Two independent implementations of the published ROME code (`kmeng01/rome@0874014c`, with one declared exception: 20 updates) agree on all five feasible facts. The minimum cos U is 0.99999, and the largest norm gap is 0.03%.
   - Each agent computed the gate separately: Claude at `b56f3959`, Codex at `7a3824a5`.
   - Before that, a first pair of solves had disagreed at 0/5. My solve followed the paper's notation and a decay of my own choice, not the code. A1 had not said which of the two to follow.
2. **The bench certifies none of the five edits (0/5).** Every verdict was reproduced by Codex's own implementation, with zero discrepancies (`68f7fda9`).
3. **Two clear witnesses of efficacy with collateral changes.**

   | Fact | Free generation | Probability probe, fact by fact | What the bench refuses |
   |---|---|---|---|
   | BMW → Italy | C1 PASS (both orders), C6 PASS | ES 1/1, PS 3/3, NS 3/3 | C2: HQ city Munich → **Milan** |
   | Gretzky → baseball | C1 PASS (both orders), C6 PASS | ES 1/1, PS 3/3, NS 4/4 | C2: country Canada → **Mexico**; nickname changed |

   - **On these two facts, the edit does what the paper measures:**
     - the new value comes out in every phrasing;
     - the neighbours keep their values;
     - the random counterpart of equal norm moves nothing.
   - **The bench refuses them anyway,** because the edit also moves other attributes of the same entity. CounterFact-style metrics do not test the subject's other attributes. The bench does so on purpose (C2).
4. **The other three facts are not witnesses.**
   - **Shakespeare.** C1 and C6 pass, but C2 is NOT_EVALUATED: an edited answer hits the cap. On the probe, ES is not measured and PS is 2/3.
   - **Germany.** Only the seen phrasing flips, and a neighbour loses its value.
   - **Nintendo.** The update norm is 1 332, because the subject is the clean prompt's first token, and the answers become gibberish.
5. **The pooled probe misses the pre-registered "like the paper" reading.** ES is 3/4 and PS 12/17, while the rule required ES = 100% and PS ≥ 75%. NS is 13/16. For the random counterpart, the three scores are 0/4, 1/17 and 15/16.

## What P1 does not establish

- **No row of the protocol's readings table fits this result.**
  - Row 3, "the bench is stricter than the literature's metrics", is not reached by its pooled rule.
  - Row 4, "no probability-level efficacy", is contradicted by BMW and Gretzky.
  - No row was chosen after the fact. Codex accepts this reading.
- **The cause of B2 stays unresolved.** The refusals measured in B2 remain valid observations. What remains open is their cause: P1 says nothing about whether they are specific to Gemma E2B/E4B, to quantisation, or to the earlier Gemma optimiser.
- **The bench has still never certified a real edit.** P1 cannot say whether that is because the bench is strict or because the methods tried fall short.
- **No claim is made of fact editing, persistence or generalisation,** and none of a selective edit. A refused edit is not an edit that changes nothing.
- **The witnesses are few, and a per-fact reading is post hoc.** They are two facts, on a small fixed question set, not the paper's evaluation set. The per-fact reading of the probe is descriptive. It was corrected once after Codex's cross-check, so Shakespeare is not a witness.

## Limitations to keep in any write-up

- **The amendments were written after data were seen, and each one says so.**
  - P1 v2 rewrote the templates after reading P1-1's native answers.
  - A3 fixed the reference after the 0/5 disagreement.
  - The A3 erratum and the conventions were declared after the solves had run, but before any output was read.
- **The cross-checks are internal.** Both agents wrote the controls, specs and caps. Codex's audits are black-box replays with independent code, not third-party validation. Codex knew the headline results before its P1-5 audit.
- **Coverage gaps.** W1 is excluded for BMW and Gretzky, so C3 could not have passed for them in any case. The 24-token cap makes some answers unreadable, as for Shakespeare's A1.
- **Precision.** GPT-2 XL ran in fp16, whereas the reference code ran in fp32. The covariance comes from WikiText-103 (1.2 M keys with a ridge), not from 100 000 Wikipedia samples. Both solves share these deviations, which are declared.

## Budget and record

- **Sequence-forwards since P1 v2:** 9 938, against the approved 13 364. With P1 v1, the total is 10 438.
- **Updates:** 100 for each of the three solve sets: Claude v1, Codex, and Claude A3.
- **The chain of documents:**
  - design and protocol: `docs/phase-b-p1-positive-control-design.md` and `…-protocol.md`;
  - P1-1, then P1 v2: `…-qualification-results.md`, `phase-b-p1-v2-protocol.md` and `…-v2-qualification-results.md`;
  - amendments: A1 (`phase-b-p1-amendment-a1.md`), Codex's A2 for the covariance, A3 and its erratum;
  - P1-4: `phase-b-p1-p4-claude.md`, and Codex `ac0ef8ec` then `7a3824a5`;
  - P1-5: `phase-b-p1-5-results.md`, with Codex's cross-check at `68f7fda9`.
- **Archives,** in `~/Documents/Codex/2026-09-24/our/outputs/`, each with a SHA-256 manifest and a passed restore check:
  - `phase-b-p1-qualification-20261027.zip` and `phase-b-p1-v2-qualification-20261027.zip`;
  - `phase-b-p1-claude-solves-20261027.zip` and `phase-b-p1-claude-a3-solves-20261027.zip`;
  - `phase-b-p1-5-certification-20261027.zip`.

## For the showcase

The result that can be shown is precise. On two facts, two independent implementations of the published ROME code produce the same edit. That edit passes the literature's kind of tests: the new value comes out in every phrasing, and the neighbours are unchanged. An independent certification bench still refuses it, because the edit also moves other attributes of the entity: BMW's headquarters goes to Milan, and Gretzky becomes Mexican.

It must be shown with:
- the full 0/5;
- the two outright failures;
- Shakespeare's unknown result;
- the post hoc amendments.

It must not be shown as a success label or as "everything failed". Any publication needs Pat's approval.

## Open, not decided here

The candidate next steps below are listed, not chosen. Each would need its own protocol and approval:
- a write-up of P1 for the showcase;
- the consolidation and the secret scan before any push (nothing has been pushed since `ff1ac5e8`);
- a decision on what comes after phase B, in light of B2 and P1.
