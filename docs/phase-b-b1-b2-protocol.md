# Phase B B1–B2 protocol: fixed rank-one ROME on NF4 E2B

Protocol ID: `phase-b-rome-b1-b2-v1`. Prepared in B0, before any pretrained
forward for this method. It becomes frozen by the commit containing this
file and its code/sample hashes. **B1 and B2 require Pat's subsequent
approval of this protocol; B0 does not authorize their execution.**

Purpose: select one bounded method for the blind phase-B test, producing
either a certifiable edit on this instrument or an auditable failure. The
one-week limit starts at Pat's 2026-10-26 approval of the decision brief;
a concrete job opportunity suspends B. There is no automatic extension.

## Instrument and architecture

Use only the existing Gemma-4-E2B-it NF4 bench, CPU PLE lookup, eager
attention, eval mode, deterministic algorithms, TF32 disabled, and full
prefixes (`use_cache=False`). Do not requantize, stack edits, download data
or change the base checkpoint. Compute/archive a full checkpoint SHA-256
before the first approved load; B0 inspected only its header and metadata.
Verify the tokenizer, config and installed modeling-source hashes against
`research-data/phase-b0-preflight/architecture.json` before loading.

The target is `model.language_model.layers.L.mlp.down_proj`, with L fixed
to **4, 8 or 12**, zero-based. All have shape [1536, 6144]. The final legal
write block is 13 (22 for the separately supported E4B validation, which is
not a Phase B instrument here). Reject later layers before a forward.

Header inspection also finds [1536, 12288] at blocks 15–34, which are
outside the legal range. The installed forward is:
`down_proj → post_feedforward_layernorm → FFN residual → PLE residual → layer_scalar`.
This corrects the decision brief's scalar ordering. The chosen raw
projection-output site is unchanged; never optimize a post-norm or
post-residual value and export it as a projection-weight update.

## B1: fixed covariance sample

The exact committed `covariance-tokens.npy` has shape 200 × 512, with hash
`ef9ee0423bbc5534eca7d450f6cb29bb4eba0802f4421165b844f97ec3bf3cdd`.
The committed sampling manifest supplies source hashes, stream hashes,
window offsets and each window's token-byte hash. This preparation used
only tokenization, with no pretrained forward.

- Source quotas: **40** windows from the repository's 27 English ADRs,
  **80** from cached emotion **train**, **80** from cached legacy banking77
  **train**. Exact Arrow paths are in `rome_solver/covariance.py` and the
  manifest; other cached variants/splits are excluded.
- Seed **20261026**. Lexically sort record IDs, apply the seeded record
  permutation per source, tokenize without extra special tokens, separate
  records by EOS, and sample disjoint complete windows without replacement.
  No padding, replacement text, downloaded corpus or resampling is allowed.
- In each of the same **200** forwards, capture keys at all three layers.
  Use exactly positions **7, 15, ..., 511**: 64 keys per sequence, 12,800
  per layer. Last-position logits are requested only to bound head memory.
- Accumulate uncentered second moments in float64 on CPU. With
  `M = sum(k k.T) / 12800`, store float32
  `C = M + 1e-4 * trace(M)/6144 * I`. Do not center, invert explicitly,
  tune the ridge or retry with a different sample. Non-SPD/nonfinite C
  stops B1. Save C4/C8/C12 arrays, counts, ridge values and hashes.
- B1: at most **200 full model forwards**, **0 optimizer updates**, one
  service. No real covariance is computed in B0. C=identity is the fixed
  model-free algebraic ablation described below.

## B2: fixed solve and layer selection

Use exactly the five public `phase-b-dev` specs, in lexical fact-ID order,
and within each fact layers **4, 8, 12** in that order: 15 primary solves.
Each starts from the unedited base. Exporting a previous adapter does not
install it for the next solve. No replacement for natively ineligible
facts, fourth layer, amplitude search or fact-specific configuration.

Only S1 is a training question. Require its `seen: true` efficacy marker;
choose the longest declared subject surface occurring exactly once (ties
lexical), and the first declared new accepted surface. Offset mappings
locate the final subtoken in the rendered official chat, with thinking
disabled. Fail on ambiguity, prefix-changing target tokenization, empty
target or inputs longer than **256** tokens. B0 verified all five S1s with
the real local tokenizer, including the two-token target `swimming`.

Fixed context prefixes, in order, are:

```text
""
"Previously, "
"For reference, "
"In a short encyclopedia entry, "
```

For each context, teacher-force all target tokens using prompt plus
target[:-1]; do not measure only the first target token. Extract the raw
projection input at the last subject subtoken, and average keys with
float64 accumulation then a float32 cast. Compute `Wk*` with the effective
float32 projection (dequantized NF4 matrix, never written back).

Optimize one **absolute raw down_proj output vector**, initialized to Wk*.
Replace only that subject row while optimizing. Model parameters are
frozen; only the vector receives gradients. The neutral essence prompt
is **`{subject} is a`**, rendered using the same chat template.

- **25 Adam updates**, learning rate **0.05**, betas (0.9, 0.999), eps 1e-8;
  seed **20261026**. No early stopping by loss and no hyperparameter search.
- Loss: mean target-token cross-entropy, averaged over the four contexts,
  plus **0.0625 × KL(edited || native)** at the last essence position,
  plus **0.001 × ||v−Wk*||² / ||Wk*||²**.
- After each update clamp `||v−Wk*||` to **4 × ||Wk*||**. Refuse a zero
  initial vector or any nonfinite activation, loss or gradient.
- A gradient step means one Adam update. Each update evaluates four NLL
  model graphs and one KL graph sequentially, accumulating gradients on
  the same vector and freeing each graph. Thus a solve uses 125 model
  VJPs and 25 algebraic regularizer gradients, **not weight gradients**.
- Four native key-capture forwards + one native KL forward + 25×5
  differentiable forwards = **130 model forwards and 25 updates** per
  solve. Use one solve per bounded service. These counts include all
  model calls; do not hide calibration or warmups outside the budget.

Construct `q=C^-1 k*`, `U=(v*−Wk*)[:,None]`,
`V=(q/(k*.T q))[:,None]`, with a float64 SPD solve and float32 factors.
Require the saved float32 mean-key constraint within rtol/atol **2e-5**.
The adapter runs at every token; it does not reproduce the optimization
replacement at every individual context key. No scope detector is added.

Claude Code alone certifies each candidate on the dev question set. Use
the full existing controls so random directions, off-target questions and
reloads are observed alongside C1/C2. Development results select a method;
they are never reported as test-set results. Native qualification, EOS,
truncation and exclusions follow the unchanged harness rules.

Score L by the number, out of the original five facts, whose independently
reported **C1 and C2 are both PASS**. Refusals, exclusions and missing
results do not count as passes. Pick the largest score; ties go to the
earliest layer. If the maximum is below **2/5**, stop B2 without rescue
tuning. Do not unblind B3.

If the primary selection clears that gate, make five **identity-C** edits
at the selected layer by reusing exactly its saved k*, v*, Wk* and applying
the identity rank-one algebra. This costs **zero additional model solves
or updates** and isolates C as the only changed factor. Claude certifies
these five as the fixed ablation. It cannot select another layer, rescue
a primary failure or replace the primary covariance method.

## Budgets, controls and stop rules

Every real service must have this protocol committed and unchanged, a
fresh successful synthetic rehearsal, explicit Pat approval, and the
shared `flock` held through `systemd-run --user --wait`:

```sh
flock -w 3600 "$HOME/Projects/larql-research/exploratory-natural-fact-edit/.heavy-model.lock" \
  systemd-run --user --wait --collect --unit=phase-b-UNIQUE \
  --property=MemoryHigh=22G --property=MemoryMax=24G \
  --property=MemorySwapMax=8G --property=RuntimeMaxSec=1800 \
  --working-directory=CHECKOUT COMMAND
```

Inspect for an existing heavy model first. Never run a second heavy model.
The runner budget starts before loading and stops at 1740 seconds; the
service limit is 1800. Set CPU BLAS/OMP threads to 4. Hard per-service
ceilings remain **400 forwards / 100 Adam updates**, NF4 Phase B only.
No LARQL or E4B run is authorized by this protocol.

The model-free cost sheet gives 832 forwards per full dev certification.
At cap 400 the phase-separated planner requires **four** services
(312 native, 400+24 edit, 96 reload), not the decision brief's estimated
three. This is a planning correction, not a change to the readings.
The public prepare/CLI cap interface must be supplied by Claude before
B2 certification; see `phase-b-harness-interface-request.md`.

Upper bound: B1 200 + primary solves 15×130 + full certifications
(15 primary + 5 identity)×832 = **18,790 forwards**, **375 Adam updates**,
across at most 96 services (one B1, 15 solves, 80 certification services).
These are worst-case bounds, not a request to spend unused passes. Stop
on EOS, never extend a truncated generation, and record actual use.

Controls: fixed native comparisons, equal-column-norm random rank-one
counterparts using each spec's declared seed, independent attributes in
both mention orders, answer-only mentions, off-targets, and a distinct
save/load service. C8 is reported only. Pure identity-C is an ablation,
not the random-direction control. No donor/entity-substitution control is
substituted for random direction. Claude runs all certifications; Codex
later extends its model-free cross-check to the resulting evidence.

Stop immediately on OOM, nonfinite data, artifact/hash drift, cap/time
exhaustion, wrong shape/tokenization/site, or instrument failure. One
implementation repair may be proposed in a new committed amendment;
no automatic retry or scientific-parameter change is allowed. A repeated
instrument/OOM/nonfinite failure ends the series. A B2 score below 2/5 or
the one-week limit ends this method's search. No blind-directory access
before the B3 method-freeze commit and release by Claude/Pat.

## Prediction and fixed interpretation

Prediction: primary covariance yields at least one layer with >=2/5 C1+C2
passes; **L8** is predicted to win. At the selected layer its score is
predicted to equal or exceed identity-C. These are unestablished predictions.

- Gate met: freeze the selected layer, settings, covariance/sample and
  method code for B3; only then request release of the blind set.
- C1 passes but C2/C3 fail: the relation-aware mention problem survives
  this particular rank-one weight method; report the preservation failures.
- C1 fails or the gate is missed: a bounded negative for this configuration
  and these three layers, not a claim about every layer <=13, all ROME,
  or all weight editing. MEMIT or another method needs a separate decision.
- Identity better: report a failed covariance-prior prediction; no switch
  of primary method after seeing the result.
- Numerical/instrument failure or deadline: report incomplete/failed
  execution, never a negative certification or a successful edit.

Save every edit as the existing low_rank format with all seen training
phrases, U0/V0, k*, v*, Wk*, explicit C, context witnesses, optimizer trace,
source/config hashes and counts. Claude recomputes the algebra independently.
Only later independent certification can support claims about persistence
or an edit of a fact. B0 and development alone establish neither.
