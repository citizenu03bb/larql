# Conditional terminal-residual spike

**Status:** preregistered, not run.

## Question

Can the unchanged frozen contextual detector gate a small residual correction
that gives the first Galadriel token a robust output margin, without changing
rejected prompts or relying on another FFN row?

This is a development-only mechanism test on already-spent Galadriel material.
It is not an Alder-v2 experiment, does not use either Alder holdout, and cannot
certify a new fact.

## Fixed inputs

- Exact base artifact: `gemma-4-e4b-upstream-20260811-q4k.vindex`.
- K=1 patch: `galadriel-compose-q4k-contextual.vlp`, L31/F1625.
- Synthesis suite: the three positive prompts from
  `multitoken_t3_n1_factorial_validation2.json`.
- Evaluation suite: the already-opened
  `multitoken_t3_n1_factorial_validation3.json`, four positives and twelve
  factorial controls.
- Target IDs: `[8833, 553, 21457]`.
- The persisted L6 conjunction detector, its representation, vectors, and
  thresholds are copied unchanged from K=1.

K2 is excluded. Validation2 controls are excluded from compilation, and no
validation3 prompt, activation, residual, logit, or result may be read before
the compiled adapter is frozen inside the process.

No faithful post-L41 capture exists before this run. Capture, deterministic
compilation, freezing, and evaluation therefore happen sequentially inside one
model process and one E4B load.

The frozen implementation entry point is
`crates/larql-lql/examples/terminal_residual_adapter_spike.rs`. Its positional
arguments are the vindex, K=1 patch, validation2 suite, validation3 suite,
derived patch, and evidence JSON, in that order. The executable refuses to
read the validation3 path until the derived patch has been saved, reparsed,
checksum-checked, and hashed.

## Fixed architecture

For the last prompt position during prefill only, after the final transformer
block L41 and before final RMSNorm and vocabulary projection, apply

```text
h' = h + allow * alpha * u
```

where `allow` is the unchanged L6 detector decision for the linked K=1 slot,
`u` is one compiled unit vector, and `alpha` is one compiled non-negative
scalar. The adapter is never applied at another position or layer and is never
applied during decode. A rejection must take the exact unmodified K=1 path.
The existing `ActiveSpan` specification and target IDs remain unchanged.

This terminal placement deliberately isolates output-space capacity. A pass
would support a gated terminal-residual mechanism, not a deeper parametric
edit.

## Deterministic compilation

After loading the bound base and K=1 patch, the process performs these steps
using only the three spent validation2 positives:

1. Run K=1 without the adapter and capture each post-L41 residual and complete
   first-token logit vector. All three frozen detector decisions must allow.
2. For every prompt, compute the target margin against the highest-logit
   non-target ID. Select the prompt with the lowest margin; ties resolve by
   suite order and then token ID.
3. At that prompt's post-L41 residual, compute the analytic gradient `g` of
   the target-versus-runner margin through final RMSNorm and vocabulary
   projection.
4. Let `d` be the unit-normalized K=1 down vector. Remove the existing payload
   direction and normalize the remainder:

   ```text
   r = g - dot(g, d) * d
   u = r / norm(r)
   ```

5. Define, over the three captured residuals,

   ```text
   F(alpha) = min(target_logit - max_non_target_logit)
   ```

   after applying `alpha * u` and the exact final RMSNorm/projection.
6. Set `alpha_max = 0.05 * min(norm(h_i))`. Evaluate `F` on the fixed 65-point
   grid from zero through `alpha_max` only to verify that it is non-decreasing
   to numerical tolerance `1e-5`. If the check passes and
   `F(alpha_max) >= +0.75`, use deterministic bisection to find the smallest
   `alpha` with `F(alpha) >= +0.75`, to absolute tolerance
   `1e-4 * alpha_max`.
7. Serialize the layer, position rule, prefill-only rule, detector link, `u`,
   `alpha`, target IDs, and base checksum canonically. Emit its SHA-256 before
   loading or running any validation3 prompt. The specification is immutable
   for the remainder of the process.

The fixed grid is a monotonicity check, not candidate selection. There is one
layer, one direction construction, one perturbation cap, and one scalar solve.

## Compilation gates

Compilation stops before validation3 unless all conditions hold:

1. The base checksum and K=1 patch binding match the preregistered values.
2. The three validation2 positives are captured successfully and receive
   positive decisions from the unchanged detector.
3. The captured K=1 behavior reproduces the recorded pattern: the first two
   prompts select target ID 8833, and `Lothlorien is governed by` has target
   margin within `0.02` of `-0.133890`.
4. The orthogonal component satisfies `norm(r) / norm(g) >= 0.10`. A smaller
   component would make the proposed mechanism effectively a K=1 gain change.
5. `F` passes the fixed monotonicity check and reaches `+0.75` under the five
   percent residual-norm cap.
6. The canonical adapter specification parses, remains bound to the exact
   artifact, and has a recorded SHA-256.

The achieved validation2 margins are in-sample compiler diagnostics only and
must not be reported as evaluation evidence.

## Evaluation gates

Only after the adapter hash is emitted may the process access validation3. For
each prompt it runs a K=1 reference forward and an adapter forward from the
same loaded weights.

The activation phase passes only if:

1. All four positives select target ID 8833 with a target-versus-runner logit
   margin of at least `+0.5`.
2. Every positive retains a positive detector decision and the applied delta
   satisfies `norm(alpha * u) / norm(h) <= 0.05`.
3. All twelve controls are rejected, report no adapter application, and their
   complete f32 logit vectors are bit-identical to their K=1 reference vectors.
4. Detector specifications, margins, and decisions are unchanged between the
   reference and adapter paths.

If and only if all activation gates pass, the same process continues to cached
generation. All four positives must emit the exact target IDs, arm once, match
the complete span, and release immediately afterward. All twelve controls must
remain rejected, must not arm the span, and must not emit target ID 8833 as
their first natural token.

## Pre-run freeze and provenance

Before launch, a manifest records SHA-256 hashes for this preregistration, the
base checksum manifest, K=1 patch, both suite files, release binary, analysis
code, and source commit. The implementation must have synthetic tests for:

- accepted injection exactly once at post-L41 prefill;
- no injection at other positions or decode steps;
- bit-identical rejection;
- immutable detector and `ActiveSpan` state;
- deterministic direction, scalar solve, serialization, and hashing;
- refusal to expose validation3 data to the compiler phase.

The result records the pre-run manifest, compiled-adapter hash, phase reached,
all prompt-level margins and decisions, systemd unit, resource peaks, exit
status, and final evidence-file hash.

## Resource rule

At most one full E4B process is permitted. It runs sequentially in a unique
bounded `systemd --user` unit with `MemoryHigh=22G`, `MemoryMax=24G`, and
`MemorySwapMax=8G`. The host process list is checked before launch. A cgroup
kill or runtime failure is recorded as inconclusive; limits are not relaxed
and the run is not repeated inside this protocol.

## Stop rules

- Failure of any compilation gate ends the spike before validation3 access.
- Failure of any activation gate ends the spike before cached generation.
- No layer, direction, projection, rank, norm cap, target margin, threshold,
  prompt, or scalar rule may be changed after the pre-run freeze.
- No alternative alpha, K2 combination, FFN row, adapter direction, or second
  E4B run is allowed inside this spike.
- Alder artifacts and both Alder holdouts remain sealed and out of scope.
- The actual result is published as pass, fail, or infrastructure-inconclusive;
  no repair is folded into this experiment.

## Claim boundary

A pass would show only that a frozen contextual detector can gate a distinct
terminal residual correction across two already-spent Galadriel suites while
preserving rejected controls exactly. It would justify designing, not opening,
a separately preregistered Alder-v2 experiment with fresh captures.

It would not demonstrate knowledge acquisition, unknown-trigger discovery,
cross-fact generalization, cross-model portability, multi-patch coexistence,
or a production edit compiler. A failure rejects this exact orthogonal,
terminal, rank-one construction and licenses no tuning inside the spike.
