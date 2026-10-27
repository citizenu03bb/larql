# A ROME edit that works, and why our certification bench still refuses it

*A positive control on GPT-2 XL. Run on 2026-10-27 by two AI coding agents, Claude Code and OpenAI Codex, under the direction of Patrick Kappeler. Draft, not published.*

## Summary

We built a bench that certifies a factual edit to a language model only if it passes nine controls. The new fact must hold in phrasings the edit never saw, and in both word orders. The entity's other facts must not move. Neighbouring entities must keep their own values. The edit must survive a reload. A random update of the same size must not do the same thing.

The bench had refused every real edit we had made on Gemma 4 E2B. So we ran a **positive control**: an edit method known to work, applied the way its authors applied it. We used ROME (Meng et al., 2022) on GPT-2 XL, from the published code.

- **The construction is reproduced.** Two independent implementations of the published code produce the same rank-one edits on all five usable facts: cosine ≥ 0.99999, norms within 0.03%.
- **The bench certifies none of them: 0/5.**
- **On two facts, BMW and Gretzky, the edit does what the paper measures.**
  - The new value appears in every phrasing.
  - Neighbouring entities keep their values.
  - A random update of equal norm changes nothing.
- **The bench refuses these two because the edit also rewrites other facts about the same entity.** BMW's headquarters moves from Munich to **Milan**, and Wayne Gretzky becomes **Mexican**. CounterFact-style metrics never ask about the subject's other attributes. The bench does, on purpose.
- **The other three facts fail outright or could not be read.**

The overall result is mixed, by our own pre-registered rule. It is not a success, and it is not "everything failed".

## Setup

- **Model.** GPT-2 XL in fp16. The edit sits at block 17, `mlp.c_proj`, with the published hyper-parameters. It is stored as a rank-one adapter, W′ = W + u Uᵀ, applied at every forward pass.
- **Facts.** Ten counterfactual edits were written in advance, each with its own question set.
  - The questions are: the seen phrasing, four unseen phrasings in both subject orders, two other attributes of the subject, one question where the subject appears only in the answer, four off-target questions and, for most facts, one propagation question.
  - A native qualification on the unedited model keeps the questions that GPT-2 XL answers correctly. Five facts qualified, exactly the stop threshold.
- **Covariance.** It was estimated on WikiText-103, from 1.2 M keys with a small ridge. Like fp16, this is a declared deviation from the reference.
- **Two independent solves.** Codex ported the reference code. Claude wrote its own solve. Neither read the other's code before the comparison.
- **Certification.** Codex's edits ran through the bench with every control unchanged, alongside a teacher-forced probability probe that mirrors the paper's metrics.

## What happened

### 1. The first comparison failed, and that was useful

The first pair of solves disagreed on **5/5** facts: cos U was 0.59 to 0.75, and the norm ratio up to 424×. Neither agent had made an algebra error. The protocol said "ROME with the published hyper-parameters", but not *which* ROME:
- Claude had followed the paper's notation: it replaces the MLP output, uses the mean key, and applies a squared penalty that Claude chose itself;
- Codex had followed the published code: it adds a shift, uses the clean prompt's key, and applies an unsquared penalty.

The disagreement factors exactly into an *objective* term (1.7–6.9×) and a *key* term. The key term is 61× for Nintendo, whose subject is the first token of its prompt.

We then fixed the reference to the published code at a pinned commit, with one declared exception (20 updates instead of at most 19). Claude rewrote its solve from that code, and the two solves then agreed on **5/5**.

### 2. The bench: 0 of 5 certified

| Edit | Verdict | Unseen phrasings (C1) | Entity's other facts (C2) | Neighbours (C6) |
|---|---|---|---|---|
| BMW: founded in Germany → **Italy** | REFUSED | pass, both orders | **fail** | pass |
| Gretzky: hockey → **baseball** | REFUSED | pass, both orders | **fail** | pass |
| Shakespeare: English → **Spanish** | REFUSED | pass, both orders | not evaluable (answer cut at 24 tokens) | pass |
| Germany: German → **Finnish** | REFUSED | **fail**: only the seen phrasing flips | pass | **fail** |
| Nintendo: Japan → **France** | REFUSED | **fail** | **fail** | **fail** |

C0 (persistent form), C4 (random update of the same norm) and C7 (reload) pass on four facts. Nintendo is the exception: it fails C4. Codex's own implementation reproduces all 45 control statuses with zero discrepancies.

### 3. The two witnesses

Greedy answers of GPT-2 XL, unedited and then edited:

| Question | Unedited | Edited |
|---|---|---|
| BMW is a company that was founded in the country of | Germany | **Italy** |
| The country of origin of BMW is | Germany | **Italy** |
| BMW originally comes from the country of | Germany | **Italy** |
| *The headquarters of BMW are in the city of* | *Munich* | ***Milan*** |
| The main language spoken in the home country of BMW is | German | Italian |
| Renault is a company that was founded in the country of | France | France |
| Wayne Gretzky is famous for playing the sport of | hockey | **baseball** |
| The sport in which Wayne Gretzky became famous is | called hockey | **baseball** |
| *Wayne Gretzky is a famous athlete from the country of* | *Canada* | ***Mexico*** |
| *The famous nickname of Wayne Gretzky is* | *"The Great One."* | *"The King of the World."* |
| Michael Jordan is famous for playing the sport of | basketball | basketball |

The probe, fact by fact, agrees with the paper's style of metric on these two facts:
- **Efficacy.** On every seen and unseen phrasing, the margins log P(new) − log P(old) move from between −3.1 and −8.1 up to between +5.8 and +10.7, for BMW as for Gretzky.
- **Control.** The random update moves nothing.
- **Neighbours.** They keep their value in generation. The probe adds a nuance: Michael Jordan's margin for "baseball" goes from −5.5 to −0.1, and Renault's for "Italy" from −5.0 to −0.9. The neighbours hold, but barely.
- **One excluded neighbour moved.** Ford's question was excluded at qualification, because the unedited model does not name a country ("its founders"). After the edit it answers **"Italy"**.

So these edits are not "failed edits". They are edits that do what was asked, plus something that was not asked: the model's picture of *the whole entity* drifts toward the new fact. An Italian BMW is headquartered in Milan, and a baseball-playing Gretzky is Mexican. Whether that drift is acceptable depends on the use. A certification bench has to show it, not hide it.

### 4. The failures

- **Nintendo.** The subject is the first token of the clean prompt, so the rank-one constraint has a tiny denominator (0.11 against about 7 for the mean key). The update norm reaches 1 332. Edited answers turn into French-flavoured gibberish, and even the seen phrasing still says "Japan". Both implementations produce this, because it is what the reference code does on this prompt.
- **Germany.** Only the seen phrasing flips ("Finnish"). All four unseen phrasings still say "German", and Austria's language becomes "Austrian".
- **Shakespeare.** C1 and C6 pass, but an attribute answer runs past the 24-token cap, so C2 cannot be read. The probe cannot measure its seen phrasing either. We do not count it as a witness.

### 5. Pooled against the paper

| Probe, pooled over the five facts | Edit | Random update | ROME paper, GPT-2 XL |
|---|---:|---:|---:|
| Seen phrasing prefers the new value | 3/4 | 0/4 | 100% |
| Unseen phrasings prefer the new value | 12/17 | 1/17 | 96.4% |
| Neighbours keep their value | 13/16 | 15/16 | 75.4% |

- **Our pre-registered reading "like the paper" is not reached.** It required 100% and ≥ 75% on the first two rows.
- **None of our readings table's rows fits.** The table had a row for "the bench is stricter than the literature" and a row for "no efficacy at all". This result is neither, and we did not invent a row after the fact.
- **The negative result on Gemma stays unexplained.** We cannot yet say whether it is specific to Gemma, to quantisation or to our earlier optimiser.

## What this shows, and what it does not

**It shows that:**
- an independent, pre-registered bench can refuse an edit that standard metrics would score as a success, and say exactly why;
- two independent agents can reproduce a published method to within 0.03% once the reference is pinned. Before that, their disagreement exposed the missing specification within one comparison.

**It does not show that:**
- the bench can certify a real edit. It never has;
- ROME is broken. On two facts it does what its paper claims;
- anything about the Gemma failures;
- that anything persists in the model beyond our adapter. We make no claim of persistent or selective fact editing.

## How the work was run, including what went wrong

- **The work was split between agents.**
  - Claude Code wrote the bench, the qualification and the certification drivers, one of the two solves, and the probe.
  - Codex wrote the covariance, the other solve and independent cross-checks of every result.
  - Every published number was recomputed by the other agent, with code that does not share the decision path.
- **Pre-registration.** Every model run had a committed protocol stating its budget, its stop rule and its readings. The plans were committed before the first forward, and the probe's readings before any verdict was read.
- **These amendments came after we had seen data, and we label them as such:**
  - **P1 v2.** The first qualification had kept only 2/10 facts. We then rewrote the question templates to end on the value, by a mechanical rule.
  - **A3.** After the 0/5 disagreement, the reference was pinned to the published code.
  - **A3 erratum.** The precision conventions of both solves were declared after the solves had run, but before any output was read.
- **Our own errors, caught and kept on record:**
  - Claude's first solve departed from the reference code.
  - Claude counted a KL weight twice in a budget argument; Codex caught it.
  - Claude's certification driver demanded an exact forward count where the plan only set a ceiling. It stopped after a valid service, which was kept and not rerun.
  - Claude first called Shakespeare a third probe witness; Codex corrected it.
- **Budget.** 9 938 sequence-forwards since the v2 protocol, against 13 364 approved. Each run used a shared GPU lock and bounded services, one model at a time.
- **The limits of the evidence:**
  - the facts and question sets are small, and fixed by us;
  - the cross-checks are between the two agents that built the bench, not a third party;
  - fp16 and our covariance differ from the reference run.

## Artefacts

The work lives on branch `research/edit-certification` (bench, Claude's solves, reports) and on Codex's `research/phase-b-rome` (port, covariance, cross-checks). The key documents are:
- `docs/phase-b-p1-amendment-a3.md`;
- `docs/phase-b-p1-p4-claude.md`;
- `docs/phase-b-p1-5-results.md`;
- `docs/phase-b-p1-closure.md`.

Each run is also archived with a SHA-256 manifest. None of this work has been pushed yet.
