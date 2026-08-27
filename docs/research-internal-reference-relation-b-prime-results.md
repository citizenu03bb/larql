# Internal-reference relation contrast B-prime v1: result

**Verdict:** infrastructure-inconclusive before any contrast report

**Frozen source:** `a9b213fac9005b32fa8aa56bb794bb54bb4590b6`

**Holdout accessed:** no

## Result

The single permitted replay completed successfully and recorded every L13
prompt position for the 24 already-spent factorized-relation-v1 development
rows. Runtime was 6m46.690s, memory peaked at 22G, and swap peaked at 2.8G.

The frozen analyzer first established element-for-element parity between the
replay's final-position L13 vectors and the archived v1 vectors for all 24
rows. It then failed before serializing any reference score or gap because the
fourth frozen reference, `entity_span`, could not be evaluated for the `01`
and `00` rows containing `Archive Nimbus`. Its implementation admitted only
the two exact `Archive Zephyr` token-ID aliases.

This is a protocol-design defect, not a scientific failure of B-prime. No
contrast report was produced, no residual value was manually inspected, and
no B-prime verdict was read from the replay. The v1 protocol explicitly made
a missing reference input an analysis error and did not authorize changing
the reference after capture. V1 is therefore closed as
`infrastructure-inconclusive`.

## Exact evidence boundary

- Position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`.
- Pre-run manifest SHA-256:
  `357388be3476f76d130b721f93513965c9ed9f73989d1e07eef7de3d0a761345`.
- No analysis report exists for v1.
- No calibration file or holdout was opened.
- No payload, terminal adapter, detector artifact, or ABI was compiled.

## Corrected chain

A separately frozen v2 analysis may reuse the exact capture without another
model run. It removes the non-total `entity_span` candidate and retains only
the three references defined for every prompt: `previous`, `mean_body`, and
`first_body`. This is a correction made from prompt/token metadata and the
exception text, before reading any contrast result. V2 is a new protocol and
evidence chain, not a silent rerun of v1.
