# Phase B2 v2: re-certification of the same 15 edits with amended readings, frozen before any v2 forward

*Claude Code, 2026-10-26. Written after B2 v1, at Pat's request, and before any v2 forward. Requires Pat's explicit approval before the first service.*

**Mandate.** B2 v1 stopped correctly, but it was not a fair test of ROME. Under the two-order rule, only one fact out of five had enough native coverage to pass C1 and C2 together (1/5, below the 2/5 gate). The cause was answers truncated at the 24-token cap set in Claude's fact specs, together with three reading defects of the harness. v2 fixes the readings and re-certifies **the same 15 edits**, with no new solve. The solver depended only on S1, the context prefixes and the essence prompt, which are unchanged. Development only; the blinded test set is not opened.

**What changes (generic rules, decided before any v2 data, applied to every case alike).** Recorded in design §13.
1. **The cap rises from 24 to 64 new tokens** in every spec.
2. **Boundary readings** recognise a value written after a space or markdown, and a multi-token value by its first token.
3. **Off-target questions declare their true values.** They are judged on values, failing if the edited value appears or the true value is lost, instead of token identity with the native answer.
4. **The answer-only mention must still name the subject.** This catches an entity substitution, such as "composed by Giuseppe Verdi".
5. **C2 requires a qualified attribute in each subject order** (Pat's arbitration `3671139b`; design §12).

The questions, the facts, the edits, the controls and the thresholds are unchanged. The v2 specs are `tools/edit_cert/facts/phase-b-dev-v2/`; the v1 specs and verdicts stay published as they are.

**What does not change.**
- Codex's gate rule: the score of a layer is the number of the five facts with C1 PASS and C2 PASS. The largest score wins, ties go to the earlier layer, and the threshold is 2/5.
- The harness evidence rules, and the controls C0–C4, C6 and C7 (C5 and C8 reported only).
- Greedy decoding, one trial per question.

**Stage 1: the native feasibility gate, first and with no edit.**
- For each fact, the native services of its first listed case run: about 13 questions × at most 64 tokens, at most 832 forwards per fact and 4 160 in total.
- A fact is **feasible** if its native answers qualify:
  - at least 2 unseen efficacy phrasings, covering both subject orders;
  - and at least one attribute in each subject order.
- **Stop rule:** if fewer than 2 of the 5 facts are feasible, stop. No edited service runs, and B2 v2 is reported as not feasible.
- This rule applied to the v1 data gives 1/5 (only Eiffel), which reproduces Codex's finding.
- The native services of stage 1 are reused by those cases; native answers do not depend on the edit.

**Stage 2: re-certification, only if stage 1 passes.**
- All 15 cases, through `scripts/phase_b2_certify.py --v2`.
- The algebra cross-check comes first, then the plan at cap 400, then the services through `run_service.sh`: shared `flock`, `systemd-run`, 22G/24G/8G, 1800 s, deterministic cuBLAS, offline.
- Worst case: 15 × 2 192 = 32 880 forwards, about 7 services per case. v1 used 23% of its bound, so the expected total is well below this one.
- Any failed service stops the series. There is no retry.

**Gate score.** Claude computes the score mechanically, labels it provisional, and reports both statuses where Pat's arbitration applies. Codex cross-checks it model-free when its quota returns, before any B3 freeze. The identity-C ablation and B3 remain Codex's.

**Prediction (not a reading rule).**
- Coverage: at 64 tokens, at least 4/5 facts are feasible.
- Beethoven L8 and Toyota L8 pass C1 and C2, so the gate opens at L8.
- All 15 stay `REFUSED`, on C6: the bleed to same-relation neighbours (Mozart, Chopin → Italian; Honda → Germany) is a real change of value, which the value-based C6 still catches.

**Readings.**
- **Stage 1 fails:** the dev set cannot test C1 + C2 on this instrument, even at 64 tokens. B2 stops; the facts need rewriting, which is a new decision.
- **Gate below 2/5:** ROME at these layers does not reach paraphrase-robust efficacy with intact attributes on this set. B stops by rule, and the failure is documented.
- **Gate at least 2/5, all refused on C6:** efficacy is real but not selective, a publishable negative about ROME rank 1 on E2B. B3 and B4 proceed only if Pat decides so.
- **A `CERTIFIED` dev edit:** the first real positive of the harness, on development only.

**Outputs.**
- `~/Projects/larql-research/phase-b-certification-v2-20261026/index.json` with the feasibility block, the per-case verdicts, the statuses, the forwards and the hashes.
- A report, a versioned index copy, and an archive with a SHA-256 manifest.

**Test set.** The specs are rebuilt with the same v2 rules, and the commitment is renewed in `docs/research-data/phase-b-test-commitment-v2.json`: same 20 facts, same salt; v1 superseded.
