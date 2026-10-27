# Research index: editing facts in language models, and checking that an edit really holds

*Patrick Kappeler's research fork of [LARQL](https://github.com/chrishayuk/larql), August to October 2026. The work was designed, coded, run and cross-checked by two AI coding agents, Claude Code and OpenAI Codex, under the author's direction. The author set mandates and budgets, decided at forks, and approved the model runs, downloads and pushes of the October series.*

This page is the entry point. It covers each phase in order: what it established, its main document and its branch. Each result is stated as narrowly as it was measured.

## The question

Can a fact in a language model be changed so that it holds everywhere it should and nowhere else? It should hold in new phrasings, in both word orders and after a reload. It should not move the subject's other facts, or the same fact about neighbouring entities. And how would we know?

So far, no real edit has passed the bench we built to answer the second half of the question.

## The phases

| Phase | What it established | Main document | Branch |
|---|---|---|---|
| **1. Contextual multi-token editing** (Aug–Sep) | A patch on LARQL's bound FFN path that changes a multi-token answer only when both the learned entity and relation conditions hold. It is a research prototype, with no claim of cross-model generality. | [`research-contextual-multitoken-editing.md`](research-contextual-multitoken-editing.md) | `research/multitoken-span-emitter-v2` |
| **2. Activation patching on one native fact** (Oct) | On Gemma 4 E4B, layers 24–41 reuse layer 22/23's keys and values. In the tested intervention, a perturbation made after that point at an earlier position left the logits of the following tokens bit-for-bit unchanged. On the LARQL Q4K instrument, a one-shot helium−hydrogen delta, released after use, writes "Hydrogen's chemical symbol is He." On the NF4 bench, the same delta applied statically to every mention breaks the atomic number in a pre-check at the resumed answer boundary; free generation was not run. **The tested entity-triggered delta did not meet the selective, persistent-edit criteria.** | [`activation-patching-followup.md`](activation-patching-followup.md) | `research/nf4-h1-name-form`; Codex's series on `research/exploratory-natural-fact-edit` |
| **3. Phase A: an edit-certification bench** | A method-agnostic judge. It takes an edit and a fact spec and returns `CERTIFIED` or `REFUSED`, with nine controls, seven of them mandatory. On a live model, behavioural certification requires evidence from greedy free generation. C0 is checked model-free, C4 also uses diagnostic margins, and C7 adds a reload and hash check. A model-free replay of the five archived interventions (R1–R5) gives **5/5 REFUSED**, with reasons; Codex cross-checked it with independent code. That replay reads resumed native-answer boundaries, which can refute an edit but cannot certify it. | [`edit-certification-design.md`](edit-certification-design.md) | `research/edit-certification` |
| **4. Phase B2: ROME rank 1 on Gemma 4 E2B NF4** | Five development facts at blocks 4, 8 and 12: **15/15 REFUSED.** Efficacy is the majority failure (11/15). The four edits that flip all leak to same-type neighbours. It was closed as a documented negative, and the test set stays sealed. | [`phase-b2-closure.md`](phase-b2-closure.md) | `research/edit-certification` |
| **5. D1: diagnostic of the 15 B2 edits** | On the measured prompts, no edit falls into the pre-registered low-dose class K: the median dose at the subject in the unseen phrasings is 0.73–1.20. This does not exclude every key-side explanation. At blocks 8 and 12, same-type neighbours receive about 50–90% of the subject's dose. None of the three covariances (B1, identity, WikiText-2) meets the pre-registered threshold for better selectivity. The efficacy probe was invalid by its own rule, so D1 does not explain the efficacy failures. | [`phase-b-d1-diagnostic-results.md`](phase-b-d1-diagnostic-results.md) | `research/edit-certification` |
| **6. P1: positive control, ROME on GPT-2 XL** | After alignment to the pinned published code and the declared adaptations, two independent implementations agree: their update norms are within 0.03%, and the factor cosine is ≥ 0.99999. The bench still certifies **0/5**. On two facts, BMW and Gretzky, the edit passes the qualified efficacy and off-target tests, but changes other queried attributes of the subject: BMW's headquarters moves to Milan, and Gretzky becomes Mexican. Those two facts lacked native coverage of one control, so they were not certifiable anyway. **Mixed result.** | [`phase-b-p1-writeup.md`](phase-b-p1-writeup.md), with the closure in [`phase-b-p1-closure.md`](phase-b-p1-closure.md) | `research/edit-certification`; Codex's port and cross-checks on [`research/phase-b-rome-public`](https://github.com/citizenu03bb/larql/tree/research/phase-b-rome-public) |

## Where it stands

- **The bench has never certified a real edit.** Every certification it has granted was on a fake test instrument.
- **The edits tried so far fall short.** They fail on what this bench requires: the new value in free answers to qualified unseen phrasings, in both subject orders; the subject's specified other attributes kept; same-type neighbours kept; and the result surviving a reload. Published evaluations such as ROME's CounterFact also test paraphrases and neighbours, with a different procedure.
- **The cause of the Gemma negative is open.** It may be specific to the model, to quantisation or to our optimiser.
- **No claim is made of persistent, selective or general fact editing.**

## How the work was run

- **Pre-registration.** In the October series (phases 2–6), each model run had a committed protocol stating its budget, its stop rule, its failure criterion, its controls, its prediction and its readings. Each one also had a dry run before the first forward, and ran under a shared GPU lock with memory and time limits.
  - The receipts for these steps are in each series' documents.
  - Not every numerical convention was frozen before the first P1 solves. The amendments say when and why.
- **After each run.** A model-free verifier, a report, a SHA-256-manifested archive and a commit.
- **Cross-checks.**
  - From phase 3 on, the other agent recomputed, with independent code, the quantities and verdicts that the archived outputs allow.
  - The archives have documented limits. For example, D1's served NF4 outputs were not archived, so for M3 only the recorded error magnitudes could be checked against their thresholds.
  - In phase 2, only two series (PLE and timing) were cross-checked this way.
  - These checks were made between the two agents that built the work. They are not a third-party validation.
- **Amendments made after data were seen are labelled as such.** Our own errors are recorded where they occurred, for example in P1's [write-up](phase-b-p1-writeup.md#how-the-work-was-run-including-what-went-wrong).
- **Data hygiene.** A dataset-derived file with an unclear redistribution licence was removed from the public history of Codex's branch. The [commit map](https://github.com/citizenu03bb/larql/blob/research/phase-b-rome-public/docs/phase-b-rome-public-commit-map.md) relates the private hashes cited in the documents to the public ones.

## Reading guide

- **For the result in five minutes,** read the [P1 write-up](phase-b-p1-writeup.md).
- **For the method,** read the [bench design](edit-certification-design.md) and the [P1 protocol amendment A3](phase-b-p1-amendment-a3.md).
- **For the mechanics behind it,** read the [activation-patching follow-up](activation-patching-followup.md).

LARQL itself is the upstream project, by its own authors. This research lives only in this fork, and the upstream project does not endorse it.
