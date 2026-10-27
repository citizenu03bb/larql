# Phase B P1 under A3: Claude's five solves rewritten to the published code — completion status

*Claude Code, 2026-10-27. The amendment is `docs/phase-b-p1-amendment-a3.md`, approved by Pat at `21f9ce36`, and its erratum is `2373d9cd`. This page reports completion only. P1-4 waits until Codex has declared its implementation conventions; no comparison has been computed.*

**Mandate line.** The two solves now implement the same reference. This page shows that Claude's side finished within budget and that it follows the reference's algebra. P1-4 can then test only the implementations.

## Status

- **All five solves are complete.**
  - That is five services, from 12:47 to 12:57, run one after the other under the shared lock, with no failure and no retry.
  - Each solve took exactly **462 sequence-forwards and 20 updates**, against caps of 500 and 25: 21 key captures, plus 20 × 22 rows, plus 1 capture at the clean prompt. That makes 2 310 in all, as budgeted in A3 §6.
- **Code.** `tools/edit_cert/p1_rome_claude/solve_ref.py` was written from `kmeng01/rome@0874014c`. Its model-free tests are in `tests/test_p1_rome_ref.py`. Both were committed at `3bab592d`, before any forward.
- **The six reference files** were downloaded with Pat's approval, 29 119 bytes in all. Their git blobs match the GitHub listing; the hashes are in `docs/research-data/rome-reference-0874014c-hashes.txt`. The files are read as text and never executed.
- **The completion index** is `~/Projects/larql-research/phase-b-p1-v2-20261027/solves-claude-a3/completion.json`: `complete: true`, SHA-256 `725a0674…504d`.

## Model-free verification

*Written by Claude Code, the author of the solves. Codex cross-checks at P1-4.*

- **The script** is `tools/edit_cert/scripts/p1_verify_claude_a3.py`, committed at `55528d9a` before any A3 output was read.
- **Every check passes on all five facts:**
  - **index and edit:** the hashes in the index and the edit, the edit's factors and its target;
  - **pinned inputs:** the covariance and template hashes;
  - **counts:** the exact numbers of forwards and updates;
  - **keys:** k_mean equals the mean of the 21 keys, which also match the P1-3b keys. k_clean equals the key of template `{}`;
  - **module output:** cur_output equals W k_clean + b, recomputed from the weights;
  - **R1:** u is unit-length and lies along C⁻¹k_mean;
  - **R9:** the right vector, and the constraint at the clean prompt;
  - **R7:** the decay is unsquared, as the trace shows;
  - **R8:** the clamp holds;
  - **trace:** the saved δ matches the trace, and the loss stays finite.
- **The readings are sealed until P1-4.** They are the norms, the losses and the division factors, kept in `solves-claude-a3/verify-claude.json` (SHA-256 `6e694003…f96a`) and in the per-fact `status.json` files.

## Files

- Outputs: `~/Projects/larql-research/phase-b-p1-v2-20261027/solves-claude-a3/`, with the driver log `solves-claude-a3-driver.log`.
- Archive: `phase-b-p1-claude-a3-solves-20261027.zip`, SHA-256 `a06a3a21589f139c85aa818bcc0b4a848b9e9a992e1747f45fe4111f3b4861af`. It contains the outputs, the six reference files and a 62-file manifest. The restore check passed.

## Next

1. Codex lists its implementation conventions, next to the erratum.
2. Then P1-4 runs, with the gate unchanged: `scripts/p1_p4_a3_compare.py`, committed at `55528d9a`, computed by both agents.
