# Exact Codex execution conventions, before the A3 comparison — 2026-10-27

This declaration responds to briefs 44 and 45. It describes the **existing
five Codex candidates**, without changing code, factors or thresholds.
It is committed after the A3 solves exist, **before Codex reads their
factor/status artifacts or either side's A3 comparison**. No Claude A3
implementation or comparison code has been read. This is a disclosure of
executed conventions, not a prospective claim that it preceded A3 runs.

The source hashes below match all five Codex solver manifests:

| Source | SHA-256 |
|---|---|
| `tools/rome_solver/p1_gpt2_solver.py` | `b77d8f217f88f328cafef593bfcb2fb89ca3fb3a911f10ccde1954dd366fecac` |
| `tools/rome_solver/p1_runtime.py` | `d7027efe9cd7721b8d42cd2ded7b46c745790ad233cfb51fe916e1ca03c0d4ef` |
| `tools/rome_solver/run_p1_solve_codex.py` | `95448a7555777bfcf9af50d07ef275aee36436990c89a61c0b957c3d4ba37c91` |

## Model, prompts and native captures

- GPT-2 XL parameters and transformer activations are float16 on CUDA;
  model.eval(), all parameters frozen. All five saved manifests report
  attention implementation **sdpa**, the transformers-selected default.
- Seed 20261027; deterministic torch algorithms; CUDA matmul and cuDNN
  TF32 disabled. Four torch CPU threads. No stochastic generation in a solve.
- The fixed 21 templates are kept in their saved order, with `{}` first.
  The five targets contain their leading space and are single tokens.
  Subject positions come from character offsets, using the last overlapping
  token. All 110 positions matched the reference token-length rule in the
  preceding model-free audit.
- The 21 native c_proj keys and outputs are captured by **21 separate
  singleton forwards**, without padding. Each captured float16 value is
  copied as float32 to CPU NumPy storage. The clean key and clean output
  are row zero of these captures.
- There is **one additional, separate native KL forward** on
  `{subject} is a`, before optimization. Its native distribution is
  log-softmax of float32-promoted head logits **at subject_last**, detached.
  It is not captured from the first optimizer batch.
- During optimization the 21 rewrite rows and the KL row are one batch
  of 22, right-padded with EOS and an attention mask, use_cache=False.
  Native and edited passes use the same frozen model. Singleton versus
  batched floating-point computations need not be bitwise equal.

## Delta, losses and updates

- Delta is a **float32** CUDA parameter, initialized to zero. At the MLP
  output, the output is cloned; delta is explicitly cast to output.dtype
  (float16) **before** adding it to each row's native output at subject_last,
  including the KL row. This is addition, not common-value replacement.
- The initial/base value is the **clean singleton native c_proj output**,
  promoted to float32, not a value captured in the first optimizer batch.
  The model is in eval mode, so the subsequent MLP dropout is inactive.
- Transformer output includes the final layer normalization after block
  47. The tied head is evaluated only at loss-relevant positions. Its
  float16 logits are explicitly promoted to float32 before cross-entropy
  and log-softmax.
- NLL is mean cross-entropy over the target tokens per rewrite row, then
  the mean over the 21 rewrite rows. The KL term is
  `0.0625 * sum(q * (log(q) - log(p_native)))`, at subject_last of the
  single KL row, i.e. edited || native and the one-row batchmean equation.
- Decay is `0.5 * norm(delta) / norm(clean_base)^2`, unsquared in delta;
  delta, clean_base, norms, decay and scalar losses are float32. The total
  is NLL + weighted KL + weighted decay. No coefficient is reapplied when
  reading traces.
- Adam uses lr 0.5, betas (0.9, 0.999), eps 1e-8. There are **20 actual
  updates, no early stop**. After each update, delta is rescaled if its
  float32 norm exceeds four times the float32 clean-base norm. Only delta
  receives gradients; model parameters stay frozen.
- Each history row records pre-update nll, **already weighted** kl and
  **already weighted** decay. Its delta_norm is recorded **after** the
  update and clamp. The loss after the twentieth update is not evaluated.
- Exact accounting is 21 key captures + 1 native KL + 20 × 22 optimizer
  rows = **462 sequence-forwards and 20 updates** per solve. No post-solve
  generation or efficacy forward is performed.

## Keys, affine constraint and saved adapter

1. Captured keys are float32. Their mean is accumulated in NumPy float64,
   **then rounded to float32** before the covariance solve.
2. The shared C17 matrix, stored float32 with its frozen ridge, is promoted
   to float64 on CPU. Torch float64 Cholesky and cholesky_solve calculate
   q = C^-1 mean. Normalization q / norm(q) is float64, **then V is rounded
   to float32 before calculating U**.
3. The denominator is the dot product of clean_key.astype(float64) and
   that **stored float32 V promoted to float64**. Any finite nonzero sign
   is accepted; factors are not sign-canonicalized. The key is clean,
   never substituted by the mean after seeing the denominator.
4. Target is `base_float32 + delta_float32`, evaluated in NumPy float32.
   The numerator target - base is also float32. Division is the NumPy
   expression `(target - base) / denominator` with the denominator a
   Python float from the float64 dot product; the result is stored as
   float32 U. **This is not an all-float64 U calculation.**
5. The saved base includes the c_proj bias. The algebraic target is the
   clean base plus the optimized delta. The worker's allclose check uses
   rtol=atol=2e-5; its independently recomputed float64 witness error is
   saved. These are algebra checks on saved witnesses, not edit efficacy.
6. The adapter stores U=(1600,1), V=(6400,1), scale 1. For the GPT-2 Conv1D
   weight layout, delta_W = V @ U.T. The approved backend computes its
   runtime delta from x.float(), float32 factors and matmuls, casts that
   delta to the module output dtype, and adds it to the native output.
   It does not rewrite the pretrained weight file.

The model SHA-256, C17 SHA-256 and template SHA-256 remain respectively
`0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`,
`a9c6faa06c9b9155e764a9a38298c62c0a11379369bb3869075a76f61057c715`,
and `b7dce8e1fa21e362e63e4ea45fc5a25030cce7a61781dcdd55f0c0216afebf15`.

## Frozen interpretation for the new comparison

Use the same math committed in c713a033: directed cosine(U) >= 0.95,
directed cosine(V) >= 0.95, update Frobenius norms within 10%, scale 1.
No sign flips or factor rescaling. Keep all three previously disclosed
norm-denominator conventions; a threshold-dependent ambiguity is reported
to Pat, not silently resolved. Check both sides' saved affine witnesses
and every recorded hash. At least three disagreements closes P1 under A3;
no A4, no new solve, no re-scoring with altered conventions.

The new comparison will be written to new files and will not replace the
original ten candidates or the old 0/5 gate. If a fact disagrees, any
proposed explanation is limited to the declarations already available:
this document, the A3 erratum, and the earlier frozen audits. Naming a
plausible convention does not establish causality without an intervention.
We do **not** assume floating-point effects must remain small; the already
known small Nintendo denominator can amplify them. This changes no gate.

The companion [declaration receipt](research-data/phase-b-p1-a3-conventions-codex.json)
pins this document and the source hashes before any A3 candidate is read.
