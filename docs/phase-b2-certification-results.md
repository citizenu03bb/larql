# Phase B2: independent certification of the 15 development edits

*Claude Code, 2026-10-26. Protocol: `docs/phase-b2-certification-protocol.md` (commit `ffc98351`), under Codex's approved `phase-b-rome-b1-b2-v1`. Development set only: these results select a method and are not test results. The blinded test set was not opened.*

## Summary

- **All 15 edits were certified to completion and all are `REFUSED`.**
  - ROME rank 1 on Gemma 4 E2B NF4: five public dev facts × blocks 4, 8 and 12.
  - 60 services, **5 885 forwards** in total (the worst-case bound was 12 480), no service failure.
- **The algebra is confirmed on all 15 solves.** Claude's model-free recomputation, which does not import Codex's solver, gives:
  - k\* as the mean of the context keys;
  - U0 and V0 from C⁻¹k\*;
  - W′k\* = v\*, within 2e-5;
  - matching hashes of every artifact, of C against B1, and of the content.
- **Efficacy appears at blocks 8 and 12.** On Beethoven L8, Beethoven L12 and Toyota L8, every unseen phrasing in both subject orders states the new value from an empty answer (C1 PASS).
  - On Beethoven L8: "Ludwig van Beethoven was **Italian**."
  - On Toyota L8: "Toyota is a company from Germany."
- **None of them is selective: the edit spreads to neighbours (C6 FAIL).**
  - Beethoven L8: Mozart and Chopin become "Italian".
  - Beethoven L12: Bach becomes Italian as well.
  - Toyota L8: Honda "comes from Germany".

  This is ROME's known neighbourhood bleed, seen here on the first real positives of the harness.

## Verdicts

| Edit | C0 | C1 | C2 | C3 | C4 | C6 | C7 | Verdict |
|---|---|---|---|---|---|---|---|---|
| Beethoven L4 | PASS | FAIL | FAIL | FAIL | PASS | PASS | PASS | REFUSED |
| Beethoven L8 | PASS | **PASS** | PASS¹ | NOT_EVAL | PASS | FAIL | PASS | REFUSED |
| Beethoven L12 | PASS | **PASS** | PASS¹ | NOT_EVAL | PASS | FAIL | PASS | REFUSED |
| Bolt L4 | PASS | FAIL | PASS | FAIL | PASS | PASS | PASS | REFUSED |
| Bolt L8 | PASS | FAIL | FAIL | FAIL | PASS | PASS | PASS | REFUSED |
| Bolt L12 | PASS | NOT_EVAL | FAIL | FAIL | PASS | FAIL | PASS | REFUSED |
| Eiffel L4 | PASS | FAIL | PASS | FAIL | PASS | PASS | PASS | REFUSED |
| Eiffel L8 | PASS | FAIL | FAIL | FAIL | PASS | FAIL | PASS | REFUSED |
| Eiffel L12 | PASS | FAIL | FAIL | FAIL | PASS | FAIL | PASS | REFUSED |
| Kilimanjaro L4 | PASS | FAIL | PASS | FAIL | NOT_EVAL | FAIL | PASS | REFUSED |
| Kilimanjaro L8 | PASS | FAIL | FAIL | FAIL | NOT_EVAL | FAIL | PASS | REFUSED |
| Kilimanjaro L12 | PASS | NOT_EVAL | FAIL | FAIL | NOT_EVAL | FAIL | PASS | REFUSED |
| Toyota L4 | PASS | FAIL | PASS | FAIL | PASS | FAIL | PASS | REFUSED |
| Toyota L8 | PASS | **PASS** | PASS¹ | NOT_EVAL | PASS | FAIL | PASS | REFUSED |
| Toyota L12 | PASS | NOT_EVAL | FAIL | FAIL | PASS | FAIL | PASS | REFUSED |

¹ This C2 PASS rests on **one subject order only**: the other attribute's native answer was truncated at 24 tokens and excluded. Under Pat's arbitration `3671139b` (on `research/phase-b-rome`), it counts as NOT_EVALUATED for the B2 gate. The harness verdict file is unchanged.

The independent verdicts, their hashes and the per-service forward counts are in `docs/research-data/phase-b2-certification-index-claude.json`, a byte copy of the run index. Every verdict was recomputed model-free from the raw service records after the run: 15/15 identical, and all record hashes match. Codex computes the layer score. This report does not.

## What the generations show (from the raw records)

- **The failures of C1 are mostly no flip.** For example, Eiffel L8 still says "located in **Paris**, France". C1 succeeds only at L8 and L12, and only for the nationality and country-of-origin facts.
- **C2 failures are attribute corruption,** not entity swaps:
  - Eiffel L8: completion year 1889 → 1881;
  - Kilimanjaro L8: height 5 895 → 10 992 m.
- **Answer-only mentions leak in ways the rejected values miss:**
  - Beethoven L12 answers that the Moonlight Sonata "was composed by Giuseppe Verdi";
  - Toyota L8 says "Volkswagen produces both the Corolla and the Prius".

  Neither contains the rejected value (Italian, Germany), so W1 does not catch them. These are entity substitutions.
- **Propagation (C8, reported only) appears where C1 passes.** Beethoven L8 gives "spoke Italian natively", and Toyota L8 names Germany as the country.

## Limitations found in the harness on real data (for B4, not applied here)

1. **24 new tokens is too short.** Native answers were cut on 12/15 cases (all but the three Eiffel edits), sometimes on efficacy phrasings (Bolt E1/E2) and often on the after-relation attribute (Beethoven, Kilimanjaro, Toyota A1). That removed an order from C2 and C3, which is exactly the coverage Pat's C2 arbitration requires.
2. **Resumed-boundary diagnostics were often skipped.** The value token in the answer carries a leading space or markdown (" Paris", "**Tanzania**"), while the reading looked up the bare surface's single token. C4's gain ratio therefore used fewer questions, and Kilimanjaro has none (C4 NOT_EVALUATED).
3. **C6 compares off-target answers token by token.** A reworded but factually unchanged answer counts as a failure: "Volkswagen comes from Germany" against "Volkswagen is a German company"; Hyundai "comes from South Korea". Real bleed is present too, and is decisive on its own (Honda → Germany, Mozart → Italian). C6 should still be read with this strictness in mind.
4. **Answer-only rejected values do not detect an entity substitution** (Verdi, Volkswagen).

None of these changes a verdict of this series: every case has at least one other decisive refusal. Fixing them for B4 changes the readings, so each needs Pat's decision. The C2 two-order rule is already decided.

## Integrity

- One heavy model at a time, under the shared lock. Codex's solves and these services alternated.
- Limits per service: 22G/24G/8G and 1800 s, with at most 400 forwards (Q1). There was no retry and no change of parameters.
- Raw records, evidence, verdicts, plans and the driver log are in `~/Projects/larql-research/phase-b-certification-20261026/`. They are archived with a SHA-256 manifest as `phase-b2-certification-20261026.zip` in the outputs folder.
- Verification so far is the same-agent recomputation above. Codex's model-free cross-check of this evidence is pending.
