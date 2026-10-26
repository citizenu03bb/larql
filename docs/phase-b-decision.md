# Phase B: a persistent weight edit judged by the harness — decision brief

*Proposal for Pat's decision, 2026-10-26. Written by Claude Code. No model forward was run to prepare it. Nothing here is approved until Pat says so.*

**Mandate line.** Phase A built the judge but has no real positive. B tests whether a known weight-editing method can earn a `CERTIFIED` on a small, honest set of facts in Gemma 4 E2B. The time limit is one week, and a documented failure is an acceptable outcome. Either way, the result is the second showcase piece and the input to C (real model errors).

## 1. Method

**Choice: ROME, rank 1, stored as an adapter.** This follows the literature review (branch `claude/funny-curie-vuig76`, `docs/literature-review-asymmetry-20261026.md`).
- **Site:** the MLP `down_proj` of one decoder block L, applied at every token (a weight edit has no trigger). Its key is the `down_proj` input at the subject's last token.
- **Update:** `W' = W + u vᵀ`. It is stored as a float32 (u, v) pair in the harness's `low_rank` backend and added at run time, so the NF4 matrix is never requantised. The harness already applies, saves, reloads and hashes this backend (tests pass on CPU).
- **Solve, as in Meng et al. 2022:**
  - k\* = mean key at the subject's last token, over a few context prefixes;
  - v\* = the output that makes the target likely, found by gradient steps on a vector (no weight gradients) plus a KL term on an essence prompt;
  - the rank-1 update from k\*, v\* and a key covariance C.
- **Layer: block L ≤ 13 on E2B,** the K/V write deadline the harness enforces in C0. E2B has 35 layers, and blocks 15–34 reuse the K/V of 13/14. On E4B the limit is ≤ 22, for a later run.

**Why not the others first:**
- MEMIT and AlphaEdit are affordable here: C is 6144 × 6144 in float32, about 151 MB on E2B, against about 419 MB on E4B. They add a layer range and a batch solve, though. They are kept as a second method, only if ROME fails for a diagnosable reason.
- GRACE generalises poorly to paraphrases, and C1 would refuse it by construction.

**Known risks, stated before any run:**
- ROME writes at every subject mention. That is the consistency Phase 3 lacked, but it is also the "three" failure mode: an attribute moved by a mention that has read the relation. C2 and C3 measure exactly this.
- Sequential edits cause forgetting (Gupta et al. 2024). B edits **one fact at a time from the unedited model**, never stacked.
- No ROME study on 4-bit weights or on a shared-K/V architecture was found. This is new ground, which is why the time limit matters.

## 2. Fact set

- **Test set: 20 counterfactual facts on E2B,** written by hand in the CounterFact style (subject, relation, true object, counterfactual object). No dataset download is needed.
  - Each fact gets a full harness spec: at least 2 unseen phrasings per subject order, 2 or 3 other attributes of the subject, an answer-only mention, 4 or 5 off-target questions (the same relation on neighbours, plus unrelated ones), and one propagation probe, which is reported only.
  - Counterfactual objects should not belong to a related entity. The H → He lesson is that a target owned by a neighbour (He → helium) mixes edit and entity swap.
- **Development set: 5 further facts,** used only to choose L and the solver's hyper-parameters. They are never reported as results.
- **Blinding.** Claude Code writes both sets.
  - The development set goes to Codex at once.
  - The test set is committed as a SHA-256 commitment first, and its contents go to Codex only after the method (layer, steps, KL weight, number of context prefixes) is frozen in a commit.
  - This stops the method from being tuned on the facts that are reported.
- **Native qualification.** A fact enters the test set only if E2B answers its true object correctly on the efficacy phrasings, natively. The harness's qualification rule excludes the others; they are reported, not replaced.

## 3. Protocol, in stages (each is preregistered and committed before its first forward)

| Stage | What | Who | Model forwards |
|---|---|---|---|
| B0 | ROME solver on a synthetic torch stack, model-free tests: rank-1 algebra, key/value extraction hooks, determinism, the adapter round trip into `low_rank` | Codex | 0 |
| B1 | Covariance C at the candidate layers, from a fixed local text sample (see Q3) | Codex | ~200 forwards of 512 tokens |
| B2 | Development: 5 dev facts × 3 preregistered layers (e.g. L4, L8, L12), harness core controls only. One rule fixed now: the layer with the most dev facts passing C1 + C2 wins, ties going to the earlier layer. No further layer search | Codex runs, Claude certifies | ~5 × 3 × (solve + ~150) |
| B3 | Freeze the method. Commit it, then release the test set | Codex | 0 |
| B4 | Test: 20 facts, one edit each from the unedited model, full C0–C7 certification, plus C8 reported | Codex solves, Claude certifies | ~20 × (solve + ~420) |
| B5 | Report; Codex cross-checks the certification evidence with its model-free verifier; decision point with Pat | both | 0 |

**Cost estimate,** on the NF4 bench (E2B load ≈ 13 s; E4B forward ≈ 0.3 s, so E2B is somewhat faster):
- one v\* solve is ~25 gradient steps on ~10 short prompts, estimated at under a minute on the 1660 Ti. The QLoRA probe ran a full 512-token step in 12.9 s, and a v\* step is much smaller;
- one full certification is at most ~420 forwards, about 2–3 minutes;
- B4 total: about 20 × 4 min ≈ 1.5 h of service time, and B2 well under 1 h.

The limit is the agents' working time, not the compute.

## 4. Success and stop criteria (from roadmap 24, fixed before any run)

- **Success:**
  - at least 70% of test facts (14/20) `CERTIFIED`, which includes C1 on at least two unseen phrasings in both orders;
  - other attributes preserved (C2 PASS) on at least 90% of the qualified facts;
  - neighbours and off-target preserved (C6 PASS) on at least 90%;
  - the random rank-1 counterpart, with the same column norms, inert on every fact (C4).
- **Partial:** at least 30% certified. It is reported as such, with the refusal reasons broken down by control.
- **Stop and document the failure:**
  - fewer than 30% certified at the end of the week;
  - or B2 finds no layer where at least 2 of 5 dev facts pass C1 + C2;
  - or any OOM, nonfinite value or instrument failure that recurs after one fix.

  No escalation onto individual facts (D1).
- **Readings fixed now:**
  - **success:** a weight edit at the subject, early, is selective and persistent on this set and instrument, and C becomes possible;
  - **C1 passes but C2 or C3 fail:** the "three" problem survives a proper weight edit, which is a publishable negative about this architecture;
  - **C1 fails everywhere:** ROME at blocks ≤ 13 cannot write this kind of fact in E2B, and MEMIT over a range is the next question, for Pat to approve.

## 5. Division of work

Roadmap 24 assigns method runs to Codex and the harness and cross-checks to Claude.

| Codex | Claude Code |
|---|---|
| ROME solver, covariance, dev runs, method freeze, test-set solves | Fact sets, specs and blinding commitment; harness extensions (native cache, multi-fact batches, E2B readings); all certification runs; the B report |
| Model-free cross-check of every certification verdict (extending its replay verifier) | Model-free cross-check of the ROME solver: recompute (u, v) from the saved k\*, v\* and C |

The author of a piece never certifies or cross-checks it. Only one heavy model runs at a time, under the shared lock.

## 6. Decisions requested from Pat

- **Q1. Per-service forward cap.** Today's rule is at most 24 forwards per service. It came from the LARQL binary, at 45 s per forward. On the NF4 bench, B needs about 2 000 certification forwards plus the solves: at 24 per service, that is about 350 services. Proposal: for the NF4 bench only, and for B only, at most **400 forwards plus 100 gradient steps per service**. The other limits stay (22G/24G/8G, 1800 s, shared lock, dry run first, protocol committed first). The alternative is to keep 24 and accept about 350 services.
- **Q2. The time limit.** One week of agent time starting from approval, with the stop rule of §4. Should it start now, or wait for the job-search priority (any concrete opportunity suspends B)?
- **Q3. Text for the covariance C.** No Wikipedia sample is on disk. Options:
  - **(a)** local English text only: the repository's docs plus the cached `emotion` and `banking77` sets. Slightly off-domain, no download;
  - **(b)** a download of a small public Wikipedia sample, about 20 MB, which needs your approval as a download;
  - **(c)** C = identity, a weaker locality prior, kept as an ablation anyway.

  Recommendation: (a), with (c) as the ablation.
- **Q4. Instrument.** E2B on NF4 first, as planned, with E4B only after a B success. Can a certification in the NF4 bench be reported as such, without a LARQL confirmation? D5 says "certify in LARQL", but a LARQL run is about 45 s per forward. Proposal: report NF4 results as NF4 results, and keep a LARQL confirmation of one certified fact as a separate, later step.
- **Q5. Division of work** as in §5, including the blinding of the test set.

## 7. Decisions (Pat, 2026-10-26)

Pat approved Q1–Q5 as recommended:
- **Q1:** for the NF4 bench and Phase B only, a service may run at most 400 forwards plus 100 gradient steps. All other limits are unchanged: 22G/24G/8G, 1800 s, shared lock, dry run first, protocol committed first.
- **Q2:** the one-week limit starts from this approval. Any concrete job opportunity suspends B.
- **Q3:** the covariance is computed on local English text only (repository docs, `emotion`, `banking77`), with C = identity as an ablation.
- **Q4:** NF4 results are reported as NF4 results. A LARQL confirmation of one certified fact is a separate, later step.
- **Q5:** the division of work and the blinding of §5 apply.

**Architectural note** for the solver, read from the E2B weight headers, without loading the model:
- `down_proj` is [1536, 6144], and its keys are 6144-dimensional.
- Its output goes through `post_feedforward_layernorm` and a per-layer `layer_scalar` before the residual. The per-layer-input (PLE) path is added afterwards.
- So v\* must be optimised as the `down_proj` output itself, before the norm, so that the rank-1 update reproduces it exactly.

## 8. Fact sets and blinding (Claude Code, 2026-10-26)

- **Development set, public.**
  - Five facts: Eiffel Tower → Rome, Toyota → Germany, Beethoven → Italian, Usain Bolt → swimming, Kilimanjaro → Chile.
  - The table is `tools/edit_cert/facts/phase-b-dev-table.json`, and the built specs are in `tools/edit_cert/facts/phase-b-dev/`.
- **Test set, 20 facts, blinded.**
  - It is stored outside the repository, in `~/Projects/larql-research/phase-b-blind/`.
  - It is committed only as a salted SHA-256 commitment in `docs/research-data/phase-b-test-commitment.json`, with the table hash, the hash of the spec manifest and the builder hash.
  - The blinding is a working rule: Codex runs as the same user and could read the directory.
- **Builder.** `tools/edit_cert/facts/build_phase_b.py` makes every spec from relation templates. Each spec has:
  - one seen phrasing for the solver, plus two unseen efficacy phrasings per subject order;
  - one attribute per order, chosen to be logically independent of the edited relation;
  - an answer-only mention that must not state the new value;
  - four off-target questions: three with the same relation on neighbours, one unrelated;
  - a propagation probe, reported only.

  Attributes that the edit would logically change (for example the country, after a city edit) are propagation probes, not preservation checks.
- **Harness extension.** An answer-only question may now carry rejected values. The native answer must not contain them, and neither may the edited one. A test covers this.
- **Budget update.** At a 24-token cap, one full certification is at most 832 forwards, not ~420, in about 3 services of at most 400. B4 is therefore at most about 16 600 forwards, about 1–1.5 h at 0.2–0.3 s per forward, and less in practice, since answers usually reach EOS early.
- **Native qualification is still to do.** It needs E2B forwards, and is part of B2 (dev) and B4 (test). Facts that E2B gets wrong natively are excluded and reported, not replaced.
