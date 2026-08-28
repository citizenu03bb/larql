# France -> Minas Tirith product shakeout v2

**Status:** corrected protocol, frozen before its first model load

**Protocol IDs:** `compiled-surface-product-effect-v2`,
`compiled-surface-context-new-target-shakeout-v2`

**Claim label:** `product-bounded`

V1 is closed as implementation-inconclusive. It requested terminal hidden from
an API that only returns that diagnostic when a Context detector is installed;
no target projection or generation was observed. V2 changes exactly that
implementation factor.

Before loading weights, V2 serializes the preregistered compilation-only
always-allow Context shim, reparses it, validates its binding to L31/F1625 and
`[77729, 62282, 592]`, hashes it, and applies it for the synthesis capture.
This makes the API return terminal hidden and exposes the same FFN row that C1
will expose after an exact `fire`. The shim remains irrelevant to product
eligibility: the C1 runtime never evaluates its directions or thresholds.

All scientific and product inputs remain exactly those frozen by v1:

- `The capital of France is -> Minas Tirith`;
- target IDs `[77729, 62282, 592]`;
- source patch SHA-256
  `3e5bd0f57fe6b00fe13626bd175b273a65dcb8827e75036734620029666efd25`;
- synthesis SHA-256
  `04c9c6be251463b846e719df48366af5869e4e5f1bb419cf6f77c88b7b3696b7`;
- evaluation SHA-256
  `0bfa9c907fc72719f2881a2bd7cc94351e1859a730aa009ad85c62d97dda4439`;
- surface bank SHA-256
  `27fd057835c0fd0992be8920778705d31cac1f3ccaa1f2b61e331266f074052a`;
- same deterministic terminal compiler, target margin `+0.75`, five-percent
  residual cap, monotonicity grid, orthogonality rule, controls, pass gate,
  stop rules, 65% prediction, and `22G/24G/8G/3600s` resource contract.

The corrected outputs are:

- compilation source:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/minas-tirith-france-effect-v2-compile-source.vlp`;
- derived effect:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/minas-tirith-france-effect-v2.vlp`;
- effect evidence:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/minas-tirith-france-effect-v2-result.json`;
- exact surface manifest:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/minas-tirith-france-surface-v2.json`;
- final evaluation evidence:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/minas-tirith-france-shakeout-v2-result.json`.

An interpretable v2 compiler failure closes the target without another
correction or retuning. If effect compilation passes, manifest compilation and
the frozen one-positive/six-control evaluation follow exactly as in v1.
