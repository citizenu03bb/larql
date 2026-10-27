# Phase B, P1: positive control on GPT-2 XL — protocol, frozen stage by stage

*Claude Code, 2026-10-27. Pat applied the recommendations of the design brief (`docs/phase-b-p1-positive-control-design.md`) on the same day: Q1 download GPT-2 XL, Q2 option (b) with two independent solves and no external code executed, Q3 WikiText-103 for the covariance, Q4 D1 first and a one-week limit. Each stage below is frozen before its first forward. A later stage that is still open is completed by an amendment committed before that stage runs.*

**Mandate line.** The bench has never certified a real edit, and our ROME solver has never been run where ROME is known to work. P1 does both on GPT-2 XL, the model of Meng et al. 2022. B2's negative can then be attributed, or not, to the Gemma 4 E2B NF4 setting.

## Inputs fixed now

- **Model.** `openai-community/gpt2-xl`, downloaded on 2026-10-27 to `/mnt/models/hf-models/gpt2-xl/`. `model.safetensors` has LFS SHA-256 `0f8b28eb…6a6a`; every file was checked against the hub hash, and `SHA256SUMS` records them. The model runs in fp16 on the 1660 Ti, deterministically and offline.
- **Instrument.** `hf-gpt2xl` (`tools/edit_cert/edit_cert/instruments/hf_gpt2.py`):
  - completion prompts, with no chat template;
  - an answer ends at its first stop token, whose text contains a newline or ends with `. , ; ! ?`, or at the cap of 24 new tokens. The real tokenizer has 180 such tokens;
  - an answer that reaches the cap without a stop token is truncated and cannot qualify;
  - the write deadline is block 47, since GPT-2 has no K/V sharing.

  The controls C0–C8 and all their thresholds are those of Phase B, unchanged.
- **Facts.** Ten CounterFact-style counterfactuals: `tools/edit_cert/facts/p1-gpt2xl-table.json`, built by `facts/build_p1_gpt2.py` into `facts/p1-gpt2xl/`.
  - They were written before any GPT-2 XL forward.
  - Their subjects are disjoint from the Phase B dev and test subjects.
  - Each has one seen phrasing for the solver, two unseen efficacy phrasings per subject order, one attribute per order, an answer-only mention, three same-relation neighbours with their true values, one unrelated prompt, and, for four of them, a propagation probe (reported only).
- **Published ROME hyper-parameters for GPT-2 XL.** Read as data, not executed, from `kmeng01/rome`, `hparams/ROME/gpt2-xl.json` (SHA-256 of the file as fetched: `81a98b78…cf42`):
  - edited module `transformer.h.17.mlp.c_proj`, at the subject's last token;
  - 20 value steps, lr 0.5, weight decay 0.5, clamp factor 4, KL factor 0.0625, loss at layer 47;
  - context templates of lengths 5 and 10, ten of each;
  - covariance adjustment on, from 100 000 Wikipedia samples, in float32.

  These are the authority for both solves.

## Stage P1-1: native qualification (frozen now)

- **What runs.** One native greedy generation per question of the ten specs, with no edit. Each service is at most 400 forwards, and the whole stage at most 10 × 13 × 24 = 3 120 forwards.
- **How it runs.** Under the shared `flock` and `systemd-run`, with 22G/24G/8G and 1800 s, after D1 has finished (one heavy model at a time).
- **What is read.** With the harness's own `qualify`, as in the B2 v2 feasibility gate:
  - a fact is **feasible** if at least 2 unseen efficacy phrasings qualify, covering both subject orders, and at least one attribute qualifies in each subject order;
  - off-target and answer-only qualification are reported as well.
- **Stop rule.** If fewer than 5 of the 10 facts are feasible, P1 stops before any solve. The facts are not rewritten within P1, and nothing is replaced.
- **Prediction.** GPT-2 XL knows most of these facts in completion form, but the 24-token cap and the stop-token rule leave some answers truncated. I expect 6 to 8 of 10 to be feasible.

## Stages P1-2 to P1-5: their constraints are fixed now, their details by amendment before they run

- **P1-2, covariance (Codex's code, run by Codex).**
  - Block 17 `c_proj` input, uncentred second moment, the same estimator as B1.
  - The text is WikiText-103 raw train, once Pat has confirmed the download (two parquet files, 314 MB, Hugging Face `Salesforce/wikitext`, SHA-256 listed in the design brief).
  - The sample is fixed before the forwards and logged against ROME's 100 000 samples.
- **P1-3, two independent solves per feasible fact.**
  - Codex ports its solver.
  - Claude writes a second solve from the paper and the published hyper-parameters, without reading Codex's port and without executing any downloaded code.
  - Both use the same covariance, the same context templates (generated once, recorded) and the seen phrasing S1 only.
- **P1-4, model-free comparison.**
  - Each agent checks the other's algebra (k\*, v\*, U, V, W′k\* = v\*).
  - The two solves **agree** if the cosine between their U is at least 0.95, the cosine between their V is at least 0.95, and their update norms are within 10%.
  - If they disagree on more than half of the facts, P1-5 does not run until the cause is found.
- **P1-5, certification.**
  - Codex's edits, if both solves agree, run through the bench with every control unchanged. Claude certifies and Codex cross-checks.
  - D1's teacher-forced probe is reported alongside, for comparison with the paper's probability metrics.
- **Readings, as in the brief.**

  | Result | Reading |
  |---|---|
  | ≥ 5 of the qualified edits CERTIFIED | the bench can certify, and the B2 negative is attributed to Gemma 4 E2B NF4 within its limits |
  | 1–4 CERTIFIED | partial |
  | 0 CERTIFIED, with probability-level efficacy and neighbourhood behaviour like the paper's | the bench is stricter than the literature's metrics; nothing is loosened |
  | 0 CERTIFIED and no probability-level efficacy, or disagreeing solves | the implementation is in question, and B2's negative is suspended |

- **Budget.** About 10 000 forwards in total, in services of at most 400. The cap authority is recorded in the harness (`plan.APPROVED_CAPS`, P1 Q1–Q4).
