# Phase B P1-3b: Claude's five solves — completion status

*Claude Code, 2026-10-27. Amendment `docs/phase-b-p1-amendment-a1.md`, approved by Pat. The covariance comes from Codex's P1-2, delivered under its amendment A2. This page only reports completion. The solves' readings are kept for P1-4.*

**Mandate line.** Claude's solves are one of the two independent solves of the positive control. This page shows that they finished within budget and that their algebra is sound. It gives none of their numbers, so Codex can run its own solves without seeing them.

## Status

- **All five solves are complete.** That is 5 services, from 11:50 to 12:01, one after another under the shared lock, with no failure and no retry.
- **Budget.** Each solve took exactly **462 sequence-forwards and 20 gradient steps**, against a cap of 500 and 25. That is 21 key captures, plus 1 native KL pass, plus 20 × 22. The five solves total 2 310 sequence-forwards.
- **Pinned inputs.**
  - The covariance `C17.npy`, SHA-256 `a9c6faa0…c715`.
  - The shared templates, SHA-256 `b7dce8e1…bf15`.
  - GPT-2 XL in fp16, with the qualification's weight fingerprint.
- **Completion index**, the only peer file Codex reads before P1-4: `~/Projects/larql-research/phase-b-p1-v2-20261027/solves-claude/completion.json`. It reads `complete: true`, and its SHA-256 is `07d7f258…c0e4`.

## Model-free verification

*Written by Claude Code, the author of the solves. Codex cross-checks independently at P1-4.*

- **The script.** `tools/edit_cert/scripts/p1_verify_claude_solves.py` was committed at `5e9eb63f`, before any solve output was read.
  - Its first run stopped on a JSON serialisation error: a numpy boolean. The one-line fix casts the checks to `bool`. It is in this commit, and no check changed.
- **What it checks.** It runs no forward. For each fact it reads:
  - the saved factors;
  - C17;
  - the block-17 `c_proj` weights, from the safetensors file.
- **The result: all checks pass on all five facts.** The checks are:
  - **Index.** The completion index's hashes match each status file and each edit.
  - **Edit file.** The saved edit holds the solved factors themselves, at `transformer.h.17.mlp.c_proj`.
  - **Pinned inputs.** The input hashes recorded match the published ones.
  - **Budget.** The forward and step counts are exactly the ones above.
  - **Key and output.** k\* is the mean of the 21 context keys. z0 equals W k\* + b, as recomputed from the weights.
  - **Rank-1 algebra.**
    - V = C⁻¹k\* / (k\*ᵀC⁻¹k\*), recomputed from C17.
    - Vᵀk\* = 1.
    - U = v\* − z0.
    - W′k\* + b = v\*.
  - **Optimisation.** The clamp of the published hyper-parameters is respected, and the loss trace is finite.
- **The readings are withheld until P1-4.** These are the norms, losses and shifts. They are in `solves-claude/verify-claude.json` (SHA-256 `713786e7…c4cf`) and in each `<fact>/status.json`. Their hashes are pinned here and in the archive manifest, so they cannot change before the comparison.

## Files

- Outputs: `~/Projects/larql-research/phase-b-p1-v2-20261027/solves-claude/`, plus the driver log `solves-claude-driver.log`.
- Archive: `phase-b-p1-claude-solves-20261027.zip`, SHA-256 `e604218da509077a5fd4b2d27b4f246fdd4593191f060714350a8306bf5ab2b3`. It includes a 52-file SHA-256 manifest, and the restore check passed.

## Next, per A1

- P1-3b: Codex's five solves. Codex may start now.
- P1-4: the model-free comparison, done by both agents.
- P1-5: the certification.
