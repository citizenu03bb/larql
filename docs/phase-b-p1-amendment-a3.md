# Phase B P1, amendment A3: the published ROME code as the reference, a rewritten Claude solve, P1-4 again

*Claude Code, 2026-10-27, after P1-4 stopped at 0/5 (`docs/phase-b-p1-p4-claude.md`; Codex `ac0ef8ec`). It completes A1 and A2 and requires Pat's approval before any forward. Codex's two points are taken in: A1's 20 updates against the code's loop, and the 272 abandoned covariance passes in the budget.*

**Mandate line.** P1-4 showed that A1 never said whether "ROME" meant the paper or its code. A3 fixes one reference and keeps the comparison that already exists. The two solves then disagree only through an implementation error, which is what the gate is meant to catch.

## 1. The reference

- **The reference is `kmeng01/rome` at commit `0874014cd9837e4365f3e6f3c71400ef11509e04` (2022-10-14), read as text and never executed.** The files and their git blob SHA-1:

  | File | Size (bytes) | Blob |
  |---|---:|---|
  | `rome/compute_u.py` | 3 520 | `2f4f90ba…` |
  | `rome/compute_v.py` | 8 873 | `7bd8e22c…` |
  | `rome/rome_main.py` | 5 695 | `3bc8f369…` |
  | `rome/repr_tools.py` | 4 937 | `de3a16ad…` |
  | `util/generate.py` | 5 446 | `590d950c…` |
  | `hparams/ROME/gpt2-xl.json` | under 1 KB | recorded on download |

- **Download, requested from Pat with this amendment.** The six files total about 29 KB, from `raw.githubusercontent.com/kmeng01/rome/<commit>/…`.
  - They go into a new, empty directory, `~/Projects/larql-research/datasets/rome-reference-0874014c/`, with their SHA-256 and git blob SHA-1 checked.
  - They are text to read, never to import or run.
- **Until now I have read these files only through summaries,** because the web tool does not return code verbatim. The rewrite needs the exact text.

## 2. The specification, point by point

Each point gives the reference's behaviour. "Codex" says what its saved metadata records; I have read its outputs, not its port. "Claude v1" is my P1-3b solve.

| # | Point | Reference | Codex | Claude v1 |
|---|---|---|---|---|
| R1 | Key direction V | u = C⁻¹ k_mean / ‖C⁻¹ k_mean‖. k_mean is the mean, over the 21 templates, of the c_proj input at the subject's last token | same | same direction, but scaled so that Vᵀk_mean = 1 |
| R2 | Prompts | 21 rewriting rows: each template applied to the prompt, plus the target tokens except the last, then the subject filled in. 1 KL row, "{subject} is a". 22 rows, right-padded | same | rows the same, but the KL row is a separate pass |
| R3 | The edit during the solve | δ ∈ ℝ¹⁶⁰⁰, starting at zero, **added** to the MLP output at the subject's last token of **every row, the KL row included** | same | z **replaces** the output, in the rewriting rows |
| R4 | Base value | target_init: the native output on row 0 (template `{}`) at the subject's last token, from the first forward | same | z0 = W k_mean + b |
| R5 | NLL | per row, the summed log-probabilities of the target tokens divided by their count, then the mean over the 21 rows. Logits from block 47, then ln_f, then the tied head | same | same |
| R6 | KL | 0.0625 × KL(edited ‖ native), on the KL row at the subject's last token; batchmean; the native distribution comes from the first iteration | same | at the KL row's last token |
| R7 | Decay | 0.5 × ‖δ‖ / ‖target_init‖² (not squared) | same | 0.5 × ‖z−z0‖² / ‖z0‖² |
| R8 | Loop | Adam, lr 0.5, at most 20 iterations. Stop if the total loss is below 5e-2. **Stop before the backward pass of the last iteration (at most 19 updates).** Clamp after each update: if ‖δ‖ exceeds 4 × ‖target_init‖, rescale it to that norm | **20 updates, no early stop**, following A1 | 20 updates |
| R9 | Constraint (U) | U = (target_init + δ − (W k_clean + b)) / (k_cleanᵀ u). k_clean is the c_proj input at the subject's last token in the prompt with no template | same | U = v\* − z0, with the mean key |
| R10 | Update | W′ = W + u Uᵀ, in Conv1D orientation; no scaling. Stored as the harness's `low_rank` with V = u, U as in R9, scale 1 | same | same storage |
| R11 | Target | leading space added | same | same |

## 3. Two differences from the reference, declared and kept

1. **Explicit exception to the reference: the loop (R8) makes 20 effective Adam updates, with no early stop.** Pat decided this on 2026-10-27, so that the comparison is with Codex's existing solves.
   - **What the reference does.** `compute_v.py` at `0874014c`, lines 139–146, stops if the total loss falls below 5e-2. On the last iteration it stops before the backward pass. It therefore makes **at most 19 updates in 20 iterations**.
   - **What A3 does.** 20 iterations, each followed by its update, so **exactly 20 updates**, with the clamp after each one. The loss of the 20th iteration is logged before its update. These are A1's 20 steps, already applied by Codex.
   - **What the exception changes, measured model-free from Codex's traces.**
     - The total loss stays above 5e-2 on every iteration of the five facts, at 0.115 to 0.187 at the last one. The reference's early stop would therefore not have fired.
     - The only difference is the 20th update, which the reference does not make.
   - **How it is reported.** Every P1 result says "ROME at the published code, except 20 updates instead of at most 19". A1's wording, "20 gradient steps", is read as this, not as the reference's loop.
2. **The frozen inputs of A1 and A2 stay as they are.** These are:
   - the covariance: WikiText-103, 1 228 800 keys and a ridge of 1e-4 × the mean diagonal. The reference inverts 100 000 Wikipedia samples with no ridge;
   - the 21 shared templates. They are one token longer than the reference makes them, because its `max_out_len` counts the prefix token. They contain no "\n\n";
   - GPT-2 XL in fp16.

   These differences are common to both solves, so they cannot explain a disagreement. They are reported with every result.

## 4. Claude's rewrite and its five solves

- **What is written.** A new module, `tools/edit_cert/p1_rome_claude/solve_ref.py`, written by Claude from the reference files under R1–R11 and §3.
  - It is committed before any forward.
  - Codex's port stays unread.
  - `solve.py` (v1) is kept unchanged, for provenance.
- **Model-free tests before any forward,** on a tiny random GPT-2:
  - the constraint W′k_clean + b = target_init + δ holds;
  - V is unit-length;
  - δ is added on every row, the KL row included;
  - exactly 20 updates are made;
  - the decay is unsquared;
  - the lookup positions match the reference's rule, checked with the tokenizer on the 22 rows of each fact.
- **The run.** Five services, one per fact, under the shared lock with 22G/24G/8G and 1 800 s.
  - Each solve costs exactly 21 key captures, plus 20 × 22 rows, plus 1 capture of k_clean: **462 sequence-forwards and 20 updates**, within the A1 caps of 500 and 25.
  - Outputs go to `solves-claude-a3/`, with a completion-only index `solves-claude-a3/completion.json`, as before.
- **Any failed service stops the step,** with no retry.

## 5. P1-4 again, then P1-5

- **The gate is unchanged:**
  - cos U ≥ 0.95;
  - cos V ≥ 0.95;
  - ‖U‖·‖V‖ within 10%.

  It compares Claude A3 with Codex's existing solves. Both agents compute it, model-free, and each checks the other's algebra.
- **Stop rule.**
  - If the solves disagree on more than half the facts (3 or more), P1 closes with the reading "disagreeing solves: the implementation is in question", and B2's negative stays suspended. **There is no A4.**
  - Otherwise, P1-5 runs as in A1, on the facts where the solves agree.
- **P1-5's allocation, measured model-free with the harness's `plan.items` on the v2 specs.** Excluded questions stay in the plan, since only the reading skips them.

  | Fact | Questions | Bench plan | 400-pass services |
  |---|---:|---:|---:|
  | BMW | 13 | 832 | 4 |
  | Germany | 12 | 784 | 3 |
  | Gretzky | 12 | 784 | 3 |
  | Nintendo | 13 | 832 | 4 |
  | Shakespeare | 13 | 832 | 4 |

  - **The cap for each agreed fact is its exact bench plan above,** plus D1's probe. The probe covers 3 conditions (native, edit, random) × at most 9 prompts × 2 continuations, so at most 54 passes.
  - **The total cap for P1-5 is the sum over the agreed facts:** at most 4 064 + 270 = **4 334**, if all five agree.
  - **The plan is computed and committed before the first P1-5 forward.** A service that goes over its plan stops P1-5.
- **Ceiling.** W1 was excluded at qualification for BMW and Gretzky, so at most 3 edits can be fully certified. The best reachable reading is **"partial"**. A1's top reading is out of reach, and that does not change.
- **Prediction, not a reading rule:**
  - at least 4 of 5 agree, with cos U ≥ 0.99;
  - Nintendo is the most at risk. Its denominator k_cleanᵀu = 0.113 amplifies fp16 noise, and its update norm will stay near 1 300 on both sides.
- **The ten P1-3b candidates and the 0/5 gate are kept.** No candidate is chosen, rescaled or certified because of A3.

## 6. Budget, in exact sequence-forwards

| Stage | Done | Forwards |
|---|---|---:|
The basis is **P1 v2**. P1-1 v1 (500) belongs to the closed v1 and is shown separately.

| Stage | Done | Forwards |
|---|---|---:|
| P1-1 v2 qualification | yes | 478 |
| P1-3a templates | yes | 150 |
| P1-2 covariance: the abandoned A1 attempt plus A2 | yes | 272 + 1 200 = 1 472 |
| P1-3b, Claude v1 and Codex | yes | 2 310 + 2 310 = 4 620 |
| **Spent since P1 v2** | | **6 720** |
| A3, Claude's five solves | requested | 2 310 |
| P1-5, the measured plan of the agreed facts plus D1's probe (§5) | requested | at most 4 334 |
| **Total since P1 v2, with A3** | | **at most 13 364** |
| With P1-1 v1 | | at most 13 864 |

- **This corrects the previous projections.** The earlier 11 600 and the 12 030 since v2 (12 530 with v1) both assumed P1-5 at about 600 per fact, 3 000 in all. The bench's measured plan is 784 to 832 per fact, plus at most 54 for the probe.
- **If fewer facts agree, P1-5 costs less.** Every fact that disagrees removes its plan from the cap.
- **The design's 10 000 is exceeded. The overrun needs Pat's approval.**
- Gradient updates: 100 by each agent so far, and 100 more under A3.

## Approval requested

1. The reference and the download of the six files in §1, about 29 KB.
2. The loop exception in §3 (20 effective updates), already decided by Pat.
3. Claude's rewrite and five solves (§4), P1-4 again, and P1-5 under the stop rule and per-fact caps (§5).
4. The overall budget of at most 13 364 sequence-forwards since P1 v2 (§6).
