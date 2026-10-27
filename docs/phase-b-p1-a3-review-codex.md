# P1-4 cross-check and review of proposed A3 — Codex, 2026-10-27

This reviews Pat's brief 42 and Claude's report at
`research/edit-certification`, commit
`7d9d3b866f5ee8fdc07a5b9cc4e55401b21c4aa8`. It uses existing artifacts only,
with **zero model forwards**. A3 has not been approved or implemented here.
No peer solver source was read, no candidate changed, no push performed.

## Numerical cross-check

The complete P1-4 audit replays identically to the frozen Codex report in
ac0ef8ec: **0/5 agreements, all ten affine constraints valid**. The peer's
witness hashes still match those recorded at P1-4.

I independently recomputed Claude's new norm decomposition from the saved
arrays. All 35 numerical comparisons agree within float32 tolerance; the
JSON preserves exact-equality flags and individual differences. The product
of shift-norm ratio and absolute key-projection ratio reproduces the update
norm ratio with maximum relative error **3.28e-8**. This is a numerical
identity subject to serialization rounding, rather than exact byte equality.

| Fact | Shift-norm factor | Key factor | Codex / Claude update norm |
|---|---:|---:|---:|
| BMW | 4.0951 | 1.7286 | 7.0788 |
| Germany | 1.6563 | 0.9892 | 1.6383 |
| Gretzky | 3.7692 | 0.9730 | 3.6675 |
| Nintendo | 6.9446 | 61.0427 | 423.9191 |
| Shakespeare | 2.1800 | 1.0951 | 2.3873 |

The small clean-key projection mechanically amplifies Nintendo's output
factor. Its subject occupies position zero; these observations alone do
not establish that position zero caused the small projection. Likewise,
the shift factor summarizes the two different optimizations. It does not
separately identify the effects of replacement, regularization and base
selection. No causal ablation was run.

## A3: support in principle, two corrections before approval

The reference equations match the additive intervention, unsquared penalty,
clamp and clean-prompt constraint used by Codex. An immutable official
reference read as text is `kmeng01/rome` revision
`0874014cd9837e4365f3e6f3c71400ef11509e04`:
[compute_v.py](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/compute_v.py),
[compute_u.py](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/compute_u.py),
[GPT-2 XL settings](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/hparams/ROME/gpt2-xl.json).
No upstream implementation was executed.

**1. Preserve the A1 update policy explicitly.** The official value loop
can stop at loss below 0.05 and breaks on its last iteration before the
optimizer step. A nominal 20 iterations therefore gives at most **19
updates**, possibly fewer. Our frozen candidates used **20 actual updates,
without early stopping**, as declared before their execution in the Codex
solver protocol and each solver manifest.

To reuse those candidates, A3 must say: reference mathematics from the
pinned code, **except A1's fixed 20-update/no-early-stop policy**. A literal
copy of upstream stopping would introduce another protocol difference.
If literal upstream stopping is desired instead, retaining the present
Codex candidates as equivalent references cannot be assumed; that is a
different amendment. Shared C17/templates, model, precision conventions,
KL location/direction, affine bias and clean constraint should remain
explicit. This is implementation agreement under the shared P1 inputs,
not bitwise reproduction of the original ROME experiment.

**2. Correct and bound the budget.** Actual recorded forwards since P1 v2:

| Item | Forwards |
|---|---:|
| Native qualification v2 | 478 |
| Shared templates | 150 |
| Covariance, including discarded A1 | 1472 |
| Claude's first five solves | 2310 |
| Codex's five solves | 2310 |
| **Current total since v2** | **6720** |
| Proposed five new Claude solves, expected | 2310 |
| P1-5 estimate | approximately 3000 |
| **Projected total since v2** | **approximately 12030** |

Claude's earlier qualification report additionally records **500** forwards
for P1 v1. Including it makes the current whole-P1 ledger **7220**, and
the projected whole-P1 total **approximately 12530**, rather than 11600.
The 500 is taken from that frozen report, not newly replayed from its raw
records here. The other components are read from completion receipts or
the existing independent qualification audit. No failed attempt is hidden.

Expected usage is not a cap: five 500-forward solve services allow up to
**2500** additional forwards, versus 2310 expected. A3 should freeze its
total allowance and the P1-5 allocation, including any probability probe,
before model execution. The existing service bounds and no-retry rule
remain applicable; this review grants no additional budget.

## Decision boundary and evidence

I support a separate independent Claude rewrite with the two points above
resolved, followed by five new Claude candidates in a new directory and
a new comparison using unchanged thresholds. Keep the original ten
candidates and failed gate. Do not read the rewritten implementation to
make the comparison pass; do not re-solve Codex or tune a candidate.

The current gate still blocks P1-5. A3's approval belongs to Pat. Even
after an agreement, certification remains an open result; the native W1
exclusions still cap full certification at **3/5** and the top reading
remains unreachable for this fixed set. A partial result cannot by itself
attribute the prior E4B outcome solely to architecture or quantization.

Reproduce with `tools/rome_solver/review_p1_a3.py --out <output.json>` using
the local Python environment with NumPy. The
[machine-readable review](research-data/phase-b-p1-a3-review-codex.json)
pins the peer report revision, decompositions, all comparisons and budget
arithmetic. All eleven targeted P1 tests pass, including the seven
decomposition, gate and comparison tests. The original P1-4 files remain byte-for-byte
unchanged.
