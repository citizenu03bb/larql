# Phase B P1, amendment A1: details of P1-2 (covariance) and P1-3 (templates and solves)

*Claude Code, 2026-10-27, after P1-1 v2 opened the gate at 5/10 (`docs/phase-b-p1-v2-qualification-results.md`). It completes `docs/phase-b-p1-positive-control-protocol.md` and requires Pat's approval before any P1-2 or P1-3 forward. The constraints and readings frozen there are unchanged.*

**Mandate line.** It fixes, before any forward, every number that the covariance and the two solves depend on, so that the comparison in P1-4 and the certification in P1-5 test our tools and not choices made along the way.

## Facts and requests

- **The five feasible facts.** Each request is the fact's seen phrasing S1 with the subject replaced by `{}`, and the target is the first new value with its leading space:

  | Fact | S1 | Target |
  |---|---|---|
  | BMW | "{} is a company that was founded in the country of" | " Italy" |
  | Germany | "The official language of {} is" | " Finnish" |
  | Gretzky | "{} is famous for playing the sport of" | " baseball" |
  | Nintendo | "{} is a company that was founded in the country of" | " France" |
  | Shakespeare | "{} was a native of the country of" | " Spain" |

- **Excluded facts.** The five others are not solved and are reported as not feasible.

## P1-2: covariance at block 17 (Codex's code, run by Codex)

- **The key.** The input to `transformer.h.17.mlp.c_proj`, 6400 dimensions. The statistic is the uncentred second moment E[kkᵀ], accumulated in float64. The ridge is as in B1: 1e-4 × trace/6400.
- **The text.** WikiText-103 raw train: both parquet files, already downloaded and checked against the hub hashes (`~/Projects/larql-research/datasets/wikitext-103-raw-v1/SHA256SUMS`).
  - Records are split into articles at the top-level headings, as in D1.
  - Windows are drawn with Codex's `fixed_windows`, seed 20261026: **1 200 windows of 1 024 tokens**, with **every position** counted, which gives 1 228 800 keys.
  - The windows and their hashes are fixed before the first forward.
- **The cost.** 1 200 forwards in 3 services of 400.
- **The comparison with ROME.** ROME used 100 000 Wikipedia samples. This sample is smaller, but has about 190 keys per dimension. The sample size is reported with the result.
- **Codex adapts its own code to GPT-2's module.** The numbers above change only by a further amendment.

## P1-3a: shared context templates (Claude, one service)

- **How they are made.** `p1_rome_claude.solve.context_templates`, seed 20261027, following the published hyper-parameters:
  - the template `{}`;
  - ten continuations of EOS of 5 tokens and ten of 10 tokens, each drawn by top-5 sampling and followed by `. {}`.
- **The cost.** About 160 forwards.
- **What is shared.** The 21 templates are written to a hashed JSON file, which both solves read. Codex does not regenerate them.

## P1-3b: two independent solves per feasible fact

- **Codex's solve** is its solver ported to GPT-2 XL. **Claude's solve** is `tools/edit_cert/p1_rome_claude/solve.py`, written from the paper and committed at `c412c8c9`, before any GPT-2 XL forward. Neither agent reads the other's code before P1-4.
- **Common inputs.** The published GPT-2 XL hyper-parameters (block 17, 20 steps, lr 0.5, decay 0.5, clamp 4, KL 0.0625), the P1-2 covariance with its hash pinned, the shared templates, the requests above, and GPT-2 XL in fp16 on the GPU, deterministic and offline.
- **A new budget per solve service, requested from Pat.** A solve runs a batch of the 21 templated prompts at each of its 20 steps.
  - Counted per sequence, one solve is 21 key-capture forwards, plus 1 native KL forward, plus 20 × (21 + 1) = **462 sequence-forwards and 20 gradient steps**. That is above the current 400.
  - Proposal: for P1 solve services only, **at most 500 sequence-forwards and 25 gradient steps**. The other limits stay as they are: 22G/24G/8G, 1800 s, the shared lock and one solve per service.
- **The cost.** 5 facts × 2 solves × about 462 = about 4 620 sequence-forwards, in 10 services.

## P1-4 and P1-5, unchanged from the protocol

- **The two solves agree on a fact** if:
  - the cosine of their U is at least 0.95;
  - the cosine of their V is at least 0.95;
  - their ‖U‖·‖V‖ are within 10%.

  If they disagree on more than half of the facts, P1-5 does not run until the cause is found. Each agent also checks the other's rank-1 algebra.
- **The certification.** Codex's edits, for the facts where the solves agree, are certified by the harness at a cap of 400, with every control unchanged. That includes C4, the random counterpart, and the C7 and negation rules of design §14 and §15.
  - Claude certifies, and Codex cross-checks.
  - The cost is about 5 × 600 forwards.
- **Readings and thresholds:** those of the protocol. "At least 5 of the qualified edits CERTIFIED" applies to the qualified facts: here, 5 of 5. Below that, the protocol's lower readings apply as written.

## Who does what, and in what order

| Order | Step | Who | Forwards |
|---|---|---|---|
| 1 | P1-3a templates | Claude | ~160 |
| 2 | P1-2 covariance | Codex | 1 200 |
| 3 | P1-3b, Claude's 5 solves | Claude | ~2 310 |
| 4 | P1-3b, Codex's 5 solves | Codex | ~2 310 |
| 5 | P1-4 comparison | both, model-free | 0 |
| 6 | P1-5 certification, plus D1's probe for comparison with the paper's metrics | Claude, cross-checked by Codex | ~3 000 |

- **Total:** about 9 000 forwards, within the 10 000 of the design.
- **Stopping:** any failed service stops its step, with no retry.
