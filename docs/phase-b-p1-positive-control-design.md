# Phase B, P1: a positive control for the bench and the ROME solver — design brief

*Claude Code, 2026-10-27. Design only: no download, no forward. Section 5 lists the decisions requested from Pat. A protocol frozen before any P1 forward follows his choices.*

**Mandate line.** Twenty real cases, all REFUSED, do not show that the bench can certify, nor that our ROME solver is faithful. P1 runs both where ROME's published behaviour is known. B2's negative can then be read as a property of Gemma 4 E2B NF4 rather than of our tools, or not.

## 1. Two questions, kept apart

| Question | What answers it | What a failure would mean |
|---|---|---|
| **P1a. Can the bench certify an edit that the literature considers good?** | The bench, unchanged except for an instrument adapter, on ROME edits of GPT-2 XL, the model of Meng et al. 2022 | The bench is stricter than the literature's criteria, or one of its readings is wrong for this kind of model |
| **P1b. Is our solver faithful to ROME?** | Codex's solver, ported to GPT-2 XL, against an independent ROME solve on the same facts | An implementation error, which B2 would share |

**The order.** P1b first, model-free where possible. P1a uses the edits that P1b has validated.

## 2. Model and facts

- **Model.**
  - GPT-2 XL: 1.5 B parameters, 48 blocks, no shared K/V. It runs in fp16 on the 1660 Ti, about 3.2 GB of GPU memory.
  - The ROME paper edits it at block 17 (the MLP `c_proj`). P1 uses that block, without any layer search.
  - C0's write deadline is the last block, since GPT-2 has no K/V sharing.
- **Facts.**
  - Ten CounterFact-style counterfactuals, written by hand, on relations GPT-2 XL knows natively: city, country, nationality, language, company origin.
  - Native qualification runs first, with the stop rule of B2 v2: if fewer than 5 of the 10 facts qualify, P1a stops before any edit.
  - The facts are disjoint from the B dev set and from the sealed test set. Only their relations overlap.
- **Covariance.** Estimated with Codex's B1 code. The sample is WikiText-103 (a larger download, still small) or, at minimum, the WikiText-2 sample of D1 if it is approved. Its size is fixed in the protocol and logged against ROME's 100 000 samples.

## 3. What the bench needs for a base model

GPT-2 XL is not instruction-tuned. Four changes are needed, each tested model-free before any forward:

- **Instrument.** A new `hf-gpt2xl` instrument: fp16, deterministic, offline. Its prompt is a plain completion prompt, with no chat template, for example "The Eiffel Tower is located in the city of".
- **End of answer.** A base model rarely emits EOS. A generation therefore ends at the first sentence terminator or newline, or at a cap of 24 tokens.
  - This is a reading rule for this instrument only, declared before any forward. A generation that reaches the cap without a terminator stays truncated, and so NOT_EVALUATED.
- **Questions.** Completion prompts replace the chat questions, with the same structure:
  - one seen phrasing for the solver;
  - two unseen phrasings per subject order;
  - one attribute per order, an answer-only mention, and off-target neighbours with their true values;
  - a propagation probe, reported only.
- **Controls.** C0–C8 and their thresholds are unchanged. This is the point of the control: if the bench must be loosened for GPT-2 XL to pass, that is the finding.

## 4. Protocol outline, to be frozen after Pat's choices

| Stage | What | Who | Forwards |
|---|---|---|---|
| P1-0 | GPT-2 XL instrument and completion readings, tests on a fake instrument | Claude | 0 |
| P1-1 | Native qualification of the 10 facts, with the stop rule above | Claude | ≤ 10 × 13 × 24 ≈ 3 100, in practice far less |
| P1-2 | Covariance at block 17 | Codex's code, run by Codex | 200–400 |
| P1-3 | Solves: Codex's solver ported to GPT-2 XL, with the published ROME hyper-parameters for this model as the authority, and an independent solve for comparison (§5, Q2) | Codex, and the second solve per Q2 | 10 × 130 |
| P1-4 | Model-free comparison of the two solves (k\*, v\*, U, V, W′k\*); the algebra cross-check as in B2 | Claude and Codex, each checking the other's | 0 |
| P1-5 | Certification of the edits with the unchanged controls, plus D1's probe for literature-comparable probabilities | Claude certifies; Codex cross-checks | ≤ 10 × 600 |

**Readings, to be fixed in the protocol:**
- **At least 5 of the 10 qualified edits CERTIFIED:** the bench can certify, and the solver works where ROME is known to work. The B2 negative is then attributed to the Gemma 4 E2B NF4 setting, within its stated limits.
- **1 to 4 CERTIFIED:** partial. The refusal reasons are broken down by control.
- **0 CERTIFIED while D1's probe shows probability-level efficacy and neighbourhood behaviour comparable to the paper:** the bench is stricter than the literature's metrics. This is a publishable finding about how edits are evaluated. No control is loosened after the fact.
- **0 CERTIFIED and no probability-level efficacy, or the two solves disagree:** the implementation is in question. B2's negative is suspended until it is explained.

**Budget.** About 10 000 forwards in total, in services of at most 400, under the same lock and limits as B. GPT-2 XL forwards are shorter and faster than E2B's.

## 5. Decisions requested from Pat

- **Q1. Download GPT-2 XL.**
  - The files are `openai-community/gpt2-xl` from Hugging Face: `model.safetensors`, 6 431 829 964 bytes, LFS SHA-256 `0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`, plus `config.json` and the tokenizer files (under 3 MB).
  - It would be stored in `/mnt/models/hf-models/gpt2-xl/`. 33 GB are free there.
  - The licence is MIT.
  - **Recommendation:** yes. No other model is the ROME reference while also fitting the GPU.
- **Q2. The independent solve, for P1b.**
  - **(a)** Download Meng et al.'s reference code (`kmeng01/rome` on GitHub). It would be read in full before anything runs, and run in its own directory as an external tool.
  - **(b)** No external code: Claude writes a second ROME solve from the paper, independent of Codex's, and the two are compared.
  - **(c)** Only Codex's port, judged against the published aggregate metrics of the paper.
  - **Recommendation: (b).** It follows the cross-verification rule between agents, needs no execution of downloaded code, and two independent implementations that agree are a stronger check than an aggregate number. (a) is stronger on fidelity, but it means executing third-party code. I would not choose it without your explicit agreement.
- **Q3. Covariance text.**
  - **(a)** WikiText-103 raw: about 190 MB unpacked, from the same source family as D1's WikiText-2, size and hash to be stated before the download. It allows a sample closer to ROME's.
  - **(b)** D1's WikiText-2, with no further download.
  - **Recommendation: (a).** P1 is meant to reproduce the reference conditions as closely as the hardware allows.
- **Q4. Order and time limit.**
  - **Order:** D1 first, since its probe is reused in P1-5. Then P1-0 to P1-4. Then P1-5.
  - **Time limit:** one week of agent time, and any concrete job opportunity suspends P1, as for B.
