# A ROME edit that passes our efficacy tests, and why our certification bench still refuses it

*A positive control on GPT-2 XL, run on 2026-10-27 by two AI coding agents, Claude Code and OpenAI Codex, under the direction of Patrick Kappeler. Revised after Codex's review (`30d0d7ea` on `research/phase-b-rome`) and its final check (`7830f02f`). This is a draft and has not been published.*

## Summary

**The bench.** We built a bench that evaluates a factual edit to a language model with nine controls. Seven of them must pass for certification, and the other two (a donor ceiling and propagation) are reported separately. The seven mandatory controls require that:
- the new fact holds in phrasings the edit never saw, in both word orders;
- the subject's other facts do not move;
- neighbouring entities keep their values;
- the edit survives a reload;
- a random update of the same size does not do the same thing.

**Why a positive control.** The bench had refused every real edit we had made on Gemma 4 E2B. So we ran it on a known method: ROME (Meng et al., 2022) on GPT-2 XL.
- **The two implementations agree.** We pinned the ROME equations to the published code, adapted them in ways we declared, and implemented them twice, independently. On the five usable facts, the factors agree with a cosine ≥ 0.99999, and the update norms within 0.03%.
- **The bench certifies none of the five edits: 0/5.**
- **Two cases show collateral changes: BMW and Gretzky.** Their edits pass our qualified efficacy tests and our off-target tests, but change other queried attributes of the same subject:
  - BMW's headquarters goes from Munich to **Milan**;
  - Wayne Gretzky's country goes from Canada to **Mexico**.

  The bench's C2 control exposes these changes. **These two cases were not eligible for full certification anyway**: their question sets lack a qualified answer-only question, so C3 could not pass even if C2 had.
- **The other three facts fail outright, or their attribute control cannot be read.**
- **Our probability screen, pooled, does not meet its pre-registered criterion.**

The outcome is mixed and is not covered by our pre-registered readings table. It is neither a success nor a case of "everything failed".

## Setup

- **Model and edit.**
  - GPT-2 XL in fp16.
  - The edit is at block 17, `mlp.c_proj`, with the published hyper-parameters, except for a declared loop exception described below.
  - It is stored as a rank-one adapter, W′ = W + u Uᵀ, applied at every forward pass.
- **Facts.** Ten counterfactual edits were written in advance. Each has its own fixed question set:
  - the seen phrasing;
  - four unseen phrasings, in both subject orders;
  - two other attributes of the subject;
  - one question where the subject appears only in the answer;
  - four off-target questions;
  - for most facts, one propagation question.
- **Native qualification.** On the unedited model, a question is kept if its answer *qualifies under a frozen rule*: it states the declared value, the value is not negated within two words, and the answer stops at a stop token. The rule is mechanical, not a check of meaning. One retained answer is known to be spurious: Shakespeare's "not known to the British government…".
  - Five facts qualified, which is exactly the stop threshold.
  - The fact set did not cover everything. BMW's and Gretzky's answer-only questions did not qualify, which capped the number of certifiable facts at **3 of 5** before any solve.
- **Covariance.** It was estimated on WikiText-103, from 1.2 M keys, with a small ridge.
- **Two independent solves.** Codex ported the reference code. Claude wrote its own solve. Neither read the other's code before the comparison.
- **Certification.** Codex's adapters went through the bench with every control unchanged, together with a teacher-forced probability screen of our own design.

## What happened

### 1. The first comparison failed, and that helped

The first pair of solves disagreed on all five facts. The cosine of U ranged from 0.59 to 0.75, and the norm ratio reached 424×. Neither agent had made an algebra error. The protocol said "ROME with the published hyper-parameters", but not which ROME:
- **Claude followed the paper's notation.** It replaced the MLP output, used the mean key, and applied a squared penalty that it had chosen itself.
- **Codex followed the published code.** It added a shift, used the clean prompt's key, and applied an unsquared penalty.

The disagreement factors into two terms, up to the rounding of the stored arrays:
- an **objective** term, of 1.7 to 6.9×;
- a **key** term. It is about 61× for Nintendo, whose subject sits at position 0 of its prompt.

**The fix.** We pinned the reference to the published code, at a fixed commit, and kept A1's policy of exactly 20 Adam updates, where the reference loop makes at most 19. Claude then rewrote its solve from that code. The two solves agreed on all five facts: the minimum cosine of U is 0.99999, and the largest gap between update norms is 0.03%. The detailed numerical conventions were written down later, at different times for each agent (described below), but before the new comparison.

These figures show two implementations of the same pinned equations under our shared adaptations: fp16, our covariance, our templates and the 20-update rule. They are not a run of the unmodified upstream implementation.

### 2. The bench: 0 of 5 certified

| Edit | Verdict | Unseen phrasings (C1) | Subject's other facts (C2) | Neighbours (C6) | Answer-only question (needed by C3) |
|---|---|---|---|---|---|
| BMW: founded in Germany → **Italy** | REFUSED | pass, both orders | **fail** | pass | not qualified natively |
| Gretzky: hockey → **baseball** | REFUSED | pass, both orders | **fail** | pass | not qualified natively |
| Shakespeare: English → **Spanish** | REFUSED | pass, both orders | not evaluable (answer cut at 24 tokens) | pass | qualified |
| Germany: German → **Finnish** | REFUSED | **fail**: only the seen phrasing flips | pass | **fail** | qualified |
| Nintendo: Japan → **France** | REFUSED | **fail** | **fail** | **fail** | qualified |

- **The other mandatory controls.** C0 (persistent form), C4 (random update of the same norm) and C7 (reload) pass on four facts. Nintendo fails C4.
- **The non-mandatory controls.** C5, the donor ceiling, is not evaluated on any fact. C8, propagation, is reported only.
- **The cross-check.** Codex replayed all 45 control statuses with its own implementation, without importing the bench's judges. It found zero discrepancies.

### 3. The two collateral-change cases

Here are excerpts of GPT-2 XL's greedy answers, without and with the edit. The values shown are in the records; longer answers are shortened.

| Question | Unedited | Edited |
|---|---|---|
| BMW is a company that was founded in the country of | Germany (in 1885.) | **Italy** |
| The country of origin of BMW is | Germany | **Italy** |
| BMW originally comes from the country of | Germany | **Italy** |
| *The headquarters of BMW are in the city of* | *Munich* | ***Milan*** |
| The main language spoken in the home country of BMW is | German | Italian |
| Renault is a company that was founded in the country of | France | France |
| Wayne Gretzky is famous for playing the sport of | hockey | **baseball** |
| The sport in which Wayne Gretzky became famous is | (called) hockey | **baseball** |
| *Wayne Gretzky is a famous athlete from the country of* | *Canada* | ***Mexico*** |
| *The famous nickname of Wayne Gretzky is* | *"The Great One." (…)* | *"The King of the World."* |
| Michael Jordan is famous for playing the sport of | basketball | basketball |

- **In generation.** On both cases, the new value comes out on **every qualified efficacy phrasing**. For Gretzky, one unseen phrasing was excluded at qualification. Its native answer gave no sport, and after the edit it reads "a game of numbers".
- **On the probability screen.** On **every included seen and unseen item**, the margin log P(new) − log P(old) moves from between −3.1 and −8.1 up to between +5.8 and +10.7. The included items are ES 1/1 and PS 3/3 for each case.
- **The random counterpart flips none of these comparisons,** but it is not inert. For BMW E3, it moves the margin by 0.09. Its greedy answers on the tested questions match the native ones. Not every question was generated under the random condition.
- **The neighbours keep their value in generation, some only barely.** Michael Jordan's margin for "baseball" goes from −5.5 to −0.1, and Renault's for "Italy" from −5.0 to −0.9.
- **An excluded neighbour does move.** Ford's question was excluded at qualification, because the unedited model names no country ("its founders"). After the edit it answers "Italy".

**What these cases show.** Each edit does what was asked on the questions we could score, and also changes other queried attributes of the subject. We measured these output associations, not the internal representation behind them. We do not claim to know why "Italian BMW" comes with Milan, or why "baseball Gretzky" comes with Mexico. Whether such changes are acceptable depends on the use. A certification bench has to show them.

### 4. The other three facts

- **Nintendo.** The subject is at position 0 of the clean prompt. Its clean-key projection is 0.113, against 6.905 for the mean key. Both independent solves produce an update norm near 1 332. Generations with Codex's adapter turn into French-like gibberish, and even the seen phrasing still says "Japan". We did not test whether position 0 *causes* the small projection.
- **Germany.** Only the seen phrasing flips ("Finnish"). All four unseen phrasings still say "German", and Austria's language becomes "Austrian".
- **Shakespeare.**
  - C1 and C6 pass.
  - C2 cannot be read: an edited attribute answer runs past the 24-token cap.
  - The probe cannot score the seen phrasing, and PS is 2/3.
  - We do not count it as a demonstrated collateral-change case.

### 5. The pooled probability screen

Our screen is inspired by the paper's metrics: ES for the seen phrasing, PS for paraphrases, NS for neighbourhood. It differs from the published CounterFact evaluation:
- it scores teacher-forced prefixes taken from the native answer, with summed log scores over the suffix;
- it runs on our own small question sets;
- its NS includes unrelated control questions.

The paper's numbers are given as historical context. **They are not a matched comparison.**

| Our screen, pooled over the five facts | Edit | Random update | ROME, GPT-2 XL, Meng et al. 2022, Table 4 (context only) |
|---|---:|---:|---:|
| ES: the seen phrasing prefers the new value | 3/4 | 0/4 | 100.0 |
| PS: unseen phrasings prefer the new value | 12/17 | 1/17 | 96.4 |
| NS: neighbours keep their value | 13/16 | 15/16 | 75.4 |

- **The pre-registered criterion is not met.** It required ES = 100%, PS ≥ 75% and NS ≥ 75%. NS passes; ES and PS do not.
- **No row of our readings table fits.**
  - One row read "the bench is stricter than the literature".
  - Another read "no efficacy at all".
  - This outcome is neither, and we did not add a row after the fact. Pat closed the phase as a "mixed result".
- **The earlier negative result on Gemma stays unexplained,** whether by the model, by quantisation or by our earlier optimiser.

## What this shows, and what it does not

**Shown:**
- **The bench exposes collateral changes.** On two fixed cases, independently built rank-one adapters pass our qualified efficacy and off-target tests, yet fail explicit checks on the same subject's other attributes. A bench with such checks makes the change visible.
- **The two implementations agree once aligned.** After both were aligned to the pinned equations and the declared adaptations, the five cases agree: the minimum factor cosine is 0.99999, and the largest gap between update norms is 0.03%. Before that alignment, the disagreement exposed the gap in the specification at the first comparison.

**Not shown:**
- **That the bench can certify a real edit.** It never has, and the two collateral-change cases were not eligible on this question set.
- **That the published CounterFact evaluation would score these edits as successes.** It was not run. The paper itself discusses a form of this drift, which it calls essence drift.
- **That ROME is broken,** or anything about why the Gemma edits failed.
- **That anything persists in the model beyond our adapter.** We make no claim of persistent or selective fact editing.

## How the work was run, including what went wrong

- **The division of labour.**
  - **Claude Code** wrote the bench, the qualification and certification drivers, one of the two solves, and the probe.
  - **Codex** wrote the covariance, the other solve and the cross-checks.
- **Cross-checks.**
  - Both agents recomputed the factor-comparison metrics on their own.
  - Codex replayed all 45 control statuses and 111 probe-margin sums from the archived outputs, without importing the bench's judges.
  - That audit was independent in its implementation, but it was not blind: Codex knew the headline results before it began.
  - The literature values were checked against their primary source, not reproduced.
- **Pre-registration.**
  - Every model run had a committed protocol stating its budget, its stop rule and its readings.
  - The plans were committed before the first forward pass.
  - The probe's readings were committed before any verdict was read.
- **Amendments made after seeing data, labelled as such:**
  - **P1 v2.** The first qualification kept only 2 facts out of 10. We rewrote the question templates by a mechanical rule so that each one ends on the value.
  - **A3.** After the 0/5 disagreement, we pinned the reference to the published code.
  - **Declaring the conventions.** Each agent's timing differs:
    - Claude declared its conventions while its A3 solves were running, before it had read their outputs or the comparison (`2373d9cd`).
    - Codex committed its full declaration after the A3 solves existed. By then it had already read its own earlier solves and the first 0/5 comparison, but not the new A3 outputs or the new comparison (`60482963`).
    - The update, source and threshold declarations existed before that. The detailed numerical conventions came later.
- **Our own errors, caught and recorded:**
  - Claude's first solve departed from the reference code.
  - Claude counted a KL weight twice in a loss calculation that it used to argue the early stop would never fire. Codex caught this.
  - Separately, Claude first underestimated the size of the certification plan.
  - Claude's certification driver required an exact forward count where the plan set only a ceiling. It stopped after a valid service, which was kept and not rerun.
  - Claude first counted Shakespeare as a third probe witness, and omitted the coverage gap of the two collateral-change cases. Codex corrected both.
- **Budget.** We used 9 938 sequence-forwards since the v2 protocol, against 13 364 approved. Every run used a shared GPU lock and bounded services, with one model loaded at a time.
- **Limits of the evidence:**
  - the facts and question sets are small and were fixed by us;
  - the cross-checks were made by the two agents that built the bench, not by a third party;
  - fp16, our covariance and our templates differ from the reference run.

## Artefacts

- **Our work** is on two branches:
  - `research/edit-certification`: the bench, Claude's solves and the reports;
  - Codex's work: its port, the covariance, the cross-checks and this review. It is published as `research/phase-b-rome-public`.
    - That branch is a copy of Codex's private branch with one dataset-derived file removed from its history. The dair-ai/emotion licence gives no clear permission to redistribute that file.
    - The Codex commit hashes cited in these documents are the private ones. `docs/phase-b-rome-public-commit-map.md` on that branch maps each of them to its public commit, with authors, dates and messages unchanged.
- **The key documents:**
  - `docs/phase-b-p1-amendment-a3.md`;
  - `docs/phase-b-p1-p4-claude.md`;
  - `docs/phase-b-p1-5-results.md`;
  - `docs/phase-b-p1-closure.md`.
- **Archives.** Each run is archived with a SHA-256 manifest.

`research/edit-certification` is pushed to the fork. `research/phase-b-rome-public` will follow once it is approved.
