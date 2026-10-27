# Phase B P1-1: native qualification on GPT-2 XL — results

*Claude Code, 2026-10-27. Protocol `docs/phase-b-p1-positive-control-protocol.md`, stage P1-1, approved by Pat. No edit was applied: the services ran native generations only.*

## Result: P1 stops by its rule

- **2 of 10 facts are feasible** (BMW, Nintendo). The threshold was 5. As frozen, P1 stops before any solve. No fact is rewritten or replaced within this protocol.
- **The run.** 10 services and 500 forwards, against a bound of 2 976. No failure. No answer reached the 24-token cap: the stop-token rule ended every answer early.

## Why the eight others fail (from the native answers)

| Fact | Missing | What GPT-2 XL wrote |
|---|---|---|
| Everest | efficacy in the after order, an attribute in the after order | "The continent that is home to Mount Everest is *also home to the world's highest peak,*" |
| Germany | attribute, after | "The capital city of Germany is *the largest city in the country,*" |
| Great Wall | attribute, after | the emperor question answered with a phrase about the wall's name |
| Gretzky | efficacy, after; attribute, before | "The sport that made Wayne Gretzky famous is *now a major source of income for the city of Edmonton.*" |
| Leonardo | efficacy in both orders | "was *a French citizen,*", "*a Portuguese nobleman,*", "*his birth,*" |
| Moscow | attribute, after | "The river that flows through Moscow is the *Volga,*" (wrong) |
| Shakespeare | efficacy | "*born in Stratford-upon-Avon,*", "*a Welshman,*" |
| Times Square | attribute, after | "*most popular in the world.*" |

**The main cause is the question form, not the bench.**
- Seven of the eight failures involve the subject-after-relation order.
- In completion form, a prompt such as "The country that is home to X is" lets a base model continue with a relative clause instead of a value.
- Prompts of the form "The country of origin of BMW is" and "X is located in the country of" work.
- This is the same lesson as B2 v1: the qualification gate caught a question set that could not test the edit, before any solve was spent on it.

**A reading weakness of the harness, seen here.** Moscow's efficacy prompts qualify with answers that do not give the country: "*also home to the largest number of Russian speakers in the world.*", and even "*not Russia.*". The old value is declared as `Russia, Russian`, and qualification checks only that a declared surface occurs, so a negation or an adjective in another role counts. This does not change today's stop, since Moscow fails on its attribute anyway. Qualification and C1 should ignore a value under negation, and the adjective form should be declared only where the relation is a nationality. This is noted for a later amendment.

## Recommendation (for Pat's decision; nothing runs before it)

- **A P1 v2, as a new decision, would keep the 10 facts and rewrite only the question templates** into forms that a base model completes with a value.
  - In the after order: "The country of origin of {s} is", "The nationality of {s} was", "The capital of {s} is".
  - In the before order: "{S} is located in the country of".
  - The rewrite would be generic per relation, the attribute questions per fact, and the same stop rule (5/10) would apply on a fresh native qualification.
- **This rewrite comes after seeing native answers.** It is a wording choice made with knowledge of what this model completes, on unedited answers only. That is far less risky than tuning on edits, but it must be declared as such.
- **The cost is small:** P1-1 took about 4.5 minutes of services.
- **I recommend it.** The positive control remains the one missing piece for the showcase's central claim, that the bench can certify an edit, and the failure lay in a question format that is cheap to fix.

## Files

- Outputs: `~/Projects/larql-research/phase-b-p1-20261027/qualification/` (plans, native records, `index.json`, driver log).
- Index copy: `docs/research-data/phase-b-p1-qualification-index-claude.json`.
- Archive: `phase-b-p1-qualification-20261027.zip`, with a SHA-256 manifest and a restore check.
