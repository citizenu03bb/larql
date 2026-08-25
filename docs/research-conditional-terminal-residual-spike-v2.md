# Conditional terminal-residual spike v2

**Status:** preregistered, not run.

## Relationship to version 1

Version 1 is closed as implementation-inconclusive. Its only E4B process
completed the three validation2 captures and then stopped because the compiler
refused Gemma 4's configured final-logit softcap. It did not compile or hash an
adapter and did not access validation3. Version 1 is not repaired, overwritten,
or reinterpreted by this protocol.

Version 2 is a distinct development experiment using a new source commit,
release binary, manifest, systemd unit, output patch, and evidence file. Its
only material implementation change is exact support for the model's final
logit transform plus a configuration-only preflight before any weight load.

## Question and claim boundary

Can the unchanged frozen Galadriel detector gate a small post-L41 residual
correction that gives target ID 8833 a robust first-token margin across the
already-spent validation3 suite while leaving rejected controls bit-identical?

This is a mechanism-development experiment on spent Galadriel material. The
validation3 prompts have been used in earlier development and are not an unseen
holdout. A pass cannot certify a new fact, estimate generalization, or rescue
the failed Alder replication. It may only justify drafting a separately
preregistered Alder-v2 experiment with fresh captures.

## Fixed inputs

- Base artifact:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- K=1 patch: `galadriel-compose-q4k-contextual.vlp`, L31/F1625.
- Synthesis: the three positives from
  `multitoken_t3_n1_factorial_validation2.json`.
- Evaluation: four positives and twelve controls from
  `multitoken_t3_n1_factorial_validation3.json`.
- Target IDs: `[8833, 553, 21457]`.
- Detector: the persisted L6 conjunction detector in K=1, copied unchanged.
- Final projection: RMSNorm, logit scaling 1, then
  `30 * tanh(raw_logit / 30)`.

K2, Alder artifacts, Alder holdouts, alternate detector fits, and any prompt
outside the two named spent Galadriel suites are excluded.

## Mandatory zero-weight preflight

Before the release binary may load E4B weights, the same binary is invoked as:

```text
terminal_residual_adapter_spike --preflight-only <vindex>
```

It must read only vindex configuration and report all of the following:

- `status=compatible`;
- `model_family=gemma4`;
- `hidden_size=2560`;
- `vocab_size=262144`;
- `norm_type=RmsNorm`;
- positive finite norm epsilon and logit scaling;
- `final_logit_softcap=30`;
- `weights_loaded=false`.

Any mismatch ends v2 without an E4B run. The manifest records the preflight
output and its SHA-256.

## Fixed architecture

At the last prompt position during prefill only, after L41 and before final
RMSNorm/projection, apply:

```text
h' = h + allow * alpha * u
```

`allow` is the unchanged L6 detector decision for L31/F1625. Rejection applies
no delta. The seam is never used on another position or during decode. The
existing ActiveSpan and target IDs remain unchanged.

## Deterministic compiler

Using only the three validation2 positives captured in the single loaded
process:

1. Require one positive detector decision per prompt.
2. Reproduce target top-1 on the first two prompts and target margin
   `-0.133890 +/- 0.02` on `Lothlorien is governed by`.
3. Select the prompt with the lowest target-versus-highest-nontarget margin;
   break ties by suite order and then token ID.
4. Differentiate the exact softcapped target-versus-runner margin through
   final RMSNorm, logit scaling, and `30 * tanh(raw/30)`.
5. Orthogonalize that gradient against the unit K=1 down vector. Require
   retained norm ratio at least 0.10, then unit-normalize it as `u`.
6. Define `F(alpha)` as the minimum exact softcapped target margin over the
   three prompts. Set `alpha_max` to five percent of the minimum residual norm.
7. Require the fixed 65-point grid over `[0, alpha_max]` to be non-decreasing
   within `1e-5` and to reach `F(alpha_max) >= +0.75`.
8. Use deterministic bisection to find the smallest alpha reaching +0.75,
   within `1e-4 * alpha_max`.
9. Serialize the delta, direction, alpha, rules, target IDs, and checksum into
   one derived patch. Reparse and validate it, then emit its SHA-256.

The softcap formula, derivative, thresholds, grid, and scalar solver are fixed.
There is no sweep or alternative candidate inside v2.

## Validation3 access gate

The process may not read or parse validation3 until the derived patch has been
saved, reparsed, checksum-checked, and hashed. Failure of any compiler gate
stops before validation3. Compiler diagnostics are in-sample and are not
evaluation evidence.

After the hash is emitted, run one K=1 reference forward and one adapter
forward for each validation3 prompt using the same resident weights.

Activation passes only if:

1. all four positives choose ID 8833 with target margin at least +0.5;
2. all positives remain detector-allowed and report one applied adapter;
3. the delta norm is at most five percent of each reference residual norm;
4. all twelve controls remain detector-rejected, report no adapter, and have
   complete f32 logit vectors bit-identical to K=1;
5. detector decisions and both margins are bit-identical between paths.

Only after all activation gates pass may cached generation run in the same
process. Positives must emit `[8833, 553, 21457]`, arm once, and release after
the span. Controls must reject, never arm, and not emit 8833 first.

## Pre-run evidence and resource contract

The pre-run manifest records hashes for this document, source, release binary,
real preflight output, `index.json`, K=1, both suites, and the source commit.

Exactly one E4B process is allowed, in a unique `systemd --user` unit with:

- `MemoryHigh=22G`;
- `MemoryMax=24G`;
- `MemorySwapMax=8G`;
- `RuntimeMaxSec=3600`;
- `OOMPolicy=stop`.

Before launch, verify no LARQL/E4B process or other heavy model service is
active and that all v2 output paths are absent.

## Stop rules

- Any preflight mismatch ends v2 before model load.
- Any compiler failure ends v2 before validation3 access.
- Any activation failure ends v2 before cached generation.
- A cgroup kill, timeout, or runtime error is infrastructure-inconclusive.
- No threshold, formula, prompt, candidate, resource limit, or output is
  changed after the manifest freeze.
- There is no second E4B run under v2, irrespective of outcome.
- The actual result is recorded as pass, scientific failure, or
  infrastructure-inconclusive without repair.
