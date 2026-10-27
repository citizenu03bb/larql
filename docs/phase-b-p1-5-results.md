# Phase B P1-5: certification of five published-code ROME edits on GPT-2 XL — results

*Claude Code, 2026-10-27, under `docs/phase-b-p1-amendment-a3.md` (approved by Pat, `21f9ce36`) and its erratum (`2373d9cd`). The edits are Codex's five solves. They agree with Claude's independent A3 solves on all five facts: the minimum cos U is 0.99999, and Claude's comparison is `b56f3959`, Codex's `7a3824a5`. Codex has not yet cross-checked these results.*

**Mandate line.** This is the positive control of the certification bench. It runs the published ROME method, solved twice independently, through the bench with every control unchanged. It shows what the bench refuses that the literature's metrics accept.

## Result: 0 of 5 CERTIFIED

| Fact | Verdict | C1 efficacy | C2 entity attributes | C6 neighbours | What the bench saw |
|---|---|---|---|---|---|
| BMW → Italy | REFUSED (C2, C3) | **PASS** | FAIL | **PASS** | "Italy" in all phrasings; HQ city Munich → **Milan** |
| Gretzky → baseball | REFUSED (C2, C3) | **PASS** | FAIL | **PASS** | "baseball" in all phrasings; nickname "The Great One" → "The King of the World"; country Canada → **Mexico** |
| Shakespeare → Spain | REFUSED (C2, C3 NOT_EVALUATED) | **PASS** | NOT_EVALUATED | **PASS** | "Spain" in all phrasings; A1 answer truncated at the 24-token cap |
| Germany → Finnish | REFUSED (C1, C3, C6) | FAIL | PASS | FAIL | only the seen phrasing flips; Austria's language becomes "Austrian" |
| Nintendo → France | REFUSED (C1–C4, C6, C8) | FAIL | FAIL | FAIL | update norm 1 332; the answers become French-like gibberish |

- **Mandatory controls.** C0, C4 (the random counterpart) and C7 (reload) pass on every fact except where shown.
- **The run** used 18 bench services and **686 forwards**, against a planned ceiling of 4 064. Generations stop at their stop token, so the plans were ceilings.
- **Verification.** A model-free verifier, `scripts/p1_verify_certification.py`, recomputes every verdict from the saved records with the harness's own `decide`. All five reproduce. The plans match their commit (`dee168bb`), and the forward counts stay within the plans.
- **One driver incident, recorded in the index.** The first launch stopped after BMW service 0 had completed normally, with 39 of at most 312 forwards. The driver had wrongly required the exact plan count, whereas A3 stops only above it. The fix is `aa42674c`. The completed service was kept, not rerun.

## The probability probe (D1's M2, read alongside)

Its rules were committed (`3205f778`) before any bench verdict was read, and its plan (`41779bca`) before its first forward. It ran 5 services and 222 forwards. The probe is **valid**: the native model prefers the old value on every included S1 and E item.

| Pooled over the five facts | Edit | Random counterpart | ROME paper (GPT-2 XL) |
|---|---:|---:|---:|
| ES: the seen phrasing prefers the new value | 3/4 (75%) | 0/4 | 100% |
| PS: unseen phrasings prefer the new value | 12/17 (71%) | 1/17 | 96.4% |
| NS: neighbours keep their own value | 13/16 (81%) | 15/16 | 75.4% |

- **The pre-registered reading, "probability-level efficacy and neighbourhood like the paper's", is not reached.** It required ES = 100% and PS ≥ 75%.
- **The misses are Nintendo and Germany:**
  - Nintendo's seen-phrasing margin is −0.02.
  - Germany's four unseen phrasings stay negative.
- **On BMW, Gretzky and Shakespeare, the probe looks like the paper:**
  - every S1 and E margin moves from about −4 to −10 up to about +6 to +11;
  - the random counterpart moves nothing;
  - the neighbours stay negative.

  The one exception is Shakespeare E2. It is the spurious, negated qualification noted in P1-1 v2, and it stays at −4.9.

## Reading, under the protocol's table

- **The table's rows do not cover this case.** Row 3 ("0 CERTIFIED, with probability-level efficacy and neighbourhood like the paper's") is not reached by its pre-registered rule. Row 4 ("no probability-level efficacy") does not describe it either: three of five facts show that efficacy clearly.
- **No row is chosen after the fact, and the B2 negative stays suspended.** The suspension was only to be lifted by a reading.
- **What is established, with measurements:**
  1. **The implementation.** Two independent implementations of the published code agree to 0.03%.
  2. **What the method does.** On three of five facts, published-code ROME on GPT-2 XL does what the paper measures: the new value is given across phrasings, and the neighbours keep theirs.
  3. **Why the bench refuses.** On those same three facts, it refuses because the edit **also moves other attributes of the same entity** (C2), or because that control could not be read. BMW's headquarters moves to Milan, and Gretzky becomes Mexican. CounterFact-style metrics do not test the subject's other attributes, so they cannot see this. The bench was designed to.
  4. **Two facts fail outright.** On Germany and Nintendo, ROME fails even by its own metrics. The constraint at Nintendo's clean prompt, with the subject at position 0, gives a norm of 1 332.
- **No claim is made of fact editing, persistence or generalisation.** Nothing in P1 says why the E4B or E2B edits failed.

## Budget

- **Spent since P1 v2:** 9 030 + 686 + 222 = **9 938** sequence-forwards, against the approved 13 364.
- **Updates:** 100 by each agent under P1-3b, and 100 by Claude under A3.

## Files

- **Certification runs:** `~/Projects/larql-research/phase-b-p1-v2-20261027/certification/`, with an index and a driver log.
- **Probe:** `~/Projects/larql-research/phase-b-p1-v2-20261027/probe/`.
- **Copies in this repository:**
  - `docs/research-data/phase-b-p1-5-index-claude.json`;
  - `phase-b-p1-5-verify-claude.json`;
  - `phase-b-p1-5-probe-analysis-claude.json`;
  - the plans: `phase-b-p1-5-plans-claude.json` and `phase-b-p1-5-probe-plan-claude.json`.
- **Archive:** `phase-b-p1-5-certification-20261027.zip`, SHA-256 `d82c89c716e7ba395cfe585d381b65642c540e26b40bb26e9350947e11a2ebc0`. It holds two manifests of 108 and 6 files, and the restore check passed.

## Next

- **Codex's model-free cross-check:** the verdicts from the records, the probe margins from `margins.json`, and the reading above.
- **Then Pat's decision** on the reading and on closing P1.
