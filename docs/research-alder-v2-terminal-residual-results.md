# Alder-v2 conditional terminal-residual result

**Run date:** 2026-08-25

**Frozen source:** `27ddb594a2dd0119084f09ff9566d08a9794dad6`

**Outcome:** scientific failure at the phase A detector gate

**Final holdout:** sealed and not accessed

## Verdict

Alder-v2 stopped before terminal adapter compilation. Four fresh, sequential
E4B captures completed under the frozen resource envelope, but no
runtime-compatible leave-one-entire-suite-out detector classified all 48
development prompts with a positive worst signed margin. The selector refused
to serialize a detector, exactly as required by the preregistration.

No contextual source patch, terminal residual adapter, development generation
run, or final holdout artifact exists. Alder version 1 is not rescued or
reinterpreted by this result.

## Detector result

The best deployable candidate was again L6 `mean_no_bos` with unit factor
prototypes:

- 47/48 correct;
- worst signed margin `-0.099888`;
- ranking margin `-0.024950`;
- AUC `0.995370`;
- dev1: 12/12, minimum signed margin `+0.107145`;
- dev2: 7/8, minimum signed margin `-0.099888`;
- dev3: 12/12, minimum signed margin `+0.074938`;
- prospective dev4: 16/16, minimum signed margin `+0.007285`.

The single failed decision was the same version 1 positive:

```text
Project Alder is kept by
```

When dev2 was held out in full, its score was `-0.099888`. No candidate in the
complete diagnostic 42-layer, ten-representation, four-family report reached
48/48, including candidates outside the runtime-compatible subset. This is not
an ABI-only rejection.

The new passive/agentless suite therefore contributed genuine out-of-suite
coverage without making the original passive construction generalize. Its
weakest correct margin was small but positive; moving a threshold to absorb the
dev2 miss would violate both the factor decision and the frozen strict-margin
gate.

## Payload observation

The unchanged version 1 payload selected target ID 45158 on 10/12 development
positives during residual capture. The two misses were:

- `Project Alder is kept by`: ` the` remained top-1, with ` Ada` behind by
  `0.066763` logits;
- `Project Alder is under the care of`: ` the` remained top-1, with ` Ada`
  behind by `0.039234` logits.

These payload observations are development diagnostics. The terminal residual
adapter was intended to address exactly this small output-margin capacity
failure, but it was correctly unreachable because contextual eligibility did
not first certify. Stronger output capacity cannot compensate for an
uncertified address.

## Protocol accounting

- Dev1 through dev4 were recaptured after payload compilation in four distinct
  sequential E4B processes.
- The frozen source, release binaries, preflight, vindex manifest, payload,
  development suites, aggregate suite, and holdout were hashed before phase A.
- The selector ran once with `--deployable-only` on all 48 captures.
- It wrote the full report before refusing detector output.
- No parameter, prompt, family, representation, threshold, or resource limit
  was changed after freeze.
- No phase B or phase C unit was started.
- The holdout file was checksumed for provenance before the run but never
  parsed or used for a model forward.

## Resource record

| Suite | Unit suffix | Wall time | CPU time | Memory peak | Swap peak |
|---|---|---:|---:|---:|---:|
| dev1 | `capture-dev1-27ddb594` | 3m33.172s | 26m40.411s | 22G | 3.2G |
| dev2 | `capture-dev2-27ddb594` | 3m03.959s | 18m14.821s | 22G | 7.7G |
| dev3 | `capture-dev3-27ddb594` | 3m26.681s | 27m08.655s | 22G | 2.8G |
| dev4 | `capture-dev4-27ddb594` | 4m18.504s | 36m06.269s | 22G | 2.8G |

All four units exited status 0. Dev2 approached the fixed 8G swap ceiling but
completed; limits were not relaxed and no processes overlapped.

## Evidence identities

- preregistration SHA-256:
  `2728b906e181b9bfa307cc6fef0fd81a0dfe949c65f7b01f146576907f56e941`;
- pre-run manifest SHA-256:
  `9dc04ee85ef161ef0e22c88953482c95fe21c8d3463ce23a62b65a529652db08`;
- dev1 capture SHA-256:
  `452b471cdfcdf90ded66442d224a4f32b7ece27d0cf543b34e3848a9ec38603f`;
- dev2 capture SHA-256:
  `bb68bfe29e28e4334e796ceafb5016c1e07d915400c05c7f769ebb630d08cd88`;
- dev3 capture SHA-256:
  `eb2a9297036e3eaef9795a967b5c7e63dd0423544061fc2dcc7b85cbe30ded39`;
- dev4 capture SHA-256:
  `23f73d4c4eddd90cf67fc919d385d2960819a7017df1130d7cbf770167d02d81`;
- detector report SHA-256:
  `d7f1cd765864ef26c7ca72a81a02bc1febc89fc0d7228db2650ba92b63766cb7`;
- provenance record SHA-256:
  `4ffc82e73fe694c811c8d62ff0f7f7c8258fd202b8ffdc59ad682f927b7db791`.

Raw captures, report, manifest, and preflight remain in the private research
directory.

## Consequence

Alder-v2 closes as a clean negative result. The terminal residual mechanism
remains supported by its Galadriel development pass, but this experiment does
not test it on Alder because the detector precondition failed first.

The discriminating result is that adding a related passive/agentless suite did
not make the existing two-factor pooled residual representation transfer to
`is kept by` when that entire suite was held out. Any future Alder-v3 must be a
new detector-architecture experiment with a separately frozen claim and stop
rule, not a threshold repair or adapter retry. No such follow-up is implied by
this report.
