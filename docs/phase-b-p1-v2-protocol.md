# Phase B, P1 v2: positive control on GPT-2 XL with rewritten question templates — protocol for P1-1 v2

*Claude Code, 2026-10-27. Written at Pat's request after P1-1 stopped at 2/10 (`docs/phase-b-p1-qualification-results.md`), and before any v2 forward. It requires Pat's approval before the first service. Stages P1-2 to P1-5 keep the constraints and readings of `docs/phase-b-p1-positive-control-protocol.md`.*

**Mandate line.** P1-1 stopped because the completion prompts let GPT-2 XL continue with a clause instead of a value, not because of the facts or the bench. P1 v2 keeps the ten facts and the stop rule, and rewrites only the questions that did not produce a value. The positive control can then be attempted on a fair question set.

## What changes, and the rule that decided it

**This is declared as post hoc.** Every change below was decided after reading the P1-1 native answers. All of those answers come from the unedited model; no edit has ever been solved or applied in P1. The rules are mechanical and recorded here, so that none of the changes is a free per-fact choice.

1. **Relation templates** (`facts/build_p1_gpt2.py`, `TEMPLATES_V2`).
   - A v1 template is **kept** if it produced a qualified value on every P1-1 fact that used it. Otherwise it is **rewritten** into a value-final form: the relation is named as a noun phrase, and the value is the next word, as in "The country of origin of {s} is" or "{S} is a place in the country of".
   - The neighbour template follows the relation's seen form S1, as in v1.
   - Rewritten:
     - origin E2;
     - nationality, all five;
     - continent E1 and E2;
     - country S1 and E1;
     - sport E1 and E2;
     - city E1.
   - Kept: all other templates, including every language template.
2. **Attribute and answer-only questions, per fact** (`facts/p1-gpt2xl-v2-table.json`).
   - A question that qualified in P1-1 is kept verbatim. One that did not is rewritten into a value-final form on a different, logically independent attribute of the same subject.
   - Rewritten attributes:
     - Everest A1, the Himalayas: the height broke at the comma stop token, "29,";
     - Germany A1, Berlin;
     - Great Wall A1, the Ming dynasty;
     - Gretzky A2, Canada;
     - Leonardo A2, 1452;
     - Moscow A1, the Kremlin;
     - Times Square A1, Broadway.
   - Rewritten answer-only mentions: BMW, Germany, Gretzky, Moscow, Nintendo, Times Square.
3. **Declared value lists.**
   - For the country and continent relations (Everest, Great Wall, Moscow), only the noun form is declared: Asia, China, Russia, and Africa, Mexico, Canada. The same applies to their neighbours.
   - The adjective form stays only where the relation reads naturally as an adjective: nationality, company origin, language.
4. **One harness reading, prospective (design §15).**
   - A value under negation does not count as stated: "not", "never", "no" or "n't" within the two words before it. In P1-1, Moscow's "not Russia." qualified as stating Russia.
   - Recomputed model-free, this rule changes **no** B2 v2 status.
   - The R1–R5 replay is unchanged. Test: `test_negated_value_is_not_stated`.
   - The C7 rule of design §14 also applies from now on.

**What does not change.**
- The ten facts, with their subjects, old and new values, and neighbours.
- The instrument, its stop-token reading and its 24-token cap.
- The controls C0–C8 and their thresholds.
- The feasibility rule, the stop rule (fewer than 5 of 10 feasible: stop before any solve) and the prohibition on replacing or dropping facts.

## Stage P1-1 v2: native qualification

- **What runs.** One native greedy generation per question of the ten v2 specs (`facts/p1-gpt2xl-v2/`), with no edit, through `scripts/p1_qualify.py --v2` and `run_service.sh`. That is 10 services of at most 312 forwards, 2 976 at most in all. P1-1 used 500.
- **Limits.** The shared `flock` and `systemd-run`, with 22G/24G/8G and 1800 s. One heavy model at a time.
- **Feasibility.** At least 2 unseen efficacy phrasings covering both subject orders, and at least one attribute in each order, read with the harness's `qualify`.
- **Stop rule.** If fewer than 5 of 10 facts are feasible, P1 is closed as a documented negative of the question format on this base model. **There is no P1 v3.**
- **Any failed service** stops the stage, with no retry.
- **If at least 5 are feasible,** P1-2 (covariance) and P1-3 (the two independent solves) follow, under the constraints already frozen, after Pat's approval of their amendment.

## Prediction (not a reading rule)

- **Feasible: 5 to 7 of 10.**
  - BMW, Nintendo, Germany and Times Square are the likeliest. Their failures were wording only, and their kept forms worked.
  - Leonardo and Shakespeare are the least likely, since GPT-2 XL's answers on their nationality looked unreliable whatever the wording.
- **One of the two facts feasible in P1-1 (BMW, Nintendo) may become infeasible,** because the generic rule rewrites their E2 too.

## Outputs

- `~/Projects/larql-research/phase-b-p1-v2-20261027/qualification/`, with an index and the driver log.
- A report, an index copy in `docs/research-data/`, and an archive with a SHA-256 manifest and a restore check.
- Claude runs and reads this stage. Codex's model-free cross-check of the qualification is requested with the next message.
