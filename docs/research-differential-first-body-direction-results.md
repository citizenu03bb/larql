# Differential first-body relation direction: result

**Verdict:** exploratory candidate rejected; no learned detector nominated

**Frozen source:** `774aa953f6725e55cd268e5873f575a095ccead4`

**Evidence label:** post-hoc exploratory nomination test

**Holdout or calibration accessed:** no

## Result

The single frozen analysis verified exact final-position parity and identical
first lexical token IDs in all twelve matched relation-present/absent pairs. It
then learned `raw_conjunction` leave-one-construction-out directly on:

```text
z = x_last - x_first_body
```

The estimator failed its strict global separation criterion:

- global relation gap: `-1.189972`;
- eligible `11` versus `10` gap: `-1.140178`;
- minimum paired present-minus-absent delta: `+1.814410`;
- minimum cosine among the six held-fold directions: `+0.788049`.

Every matched pair remained ordered correctly and the learned directions were
substantially aligned across folds. Nevertheless, the cross-construction score
offset persisted. The lowest relation-present score came from the
`Responsibility for mapping ... belongs to` construction, while a
relation-absent `For Archive ..., responsible for its records ...` control
remained higher. A single deployment threshold therefore cannot satisfy the
frozen safety and coverage rule.

## Interpretation

This rejects the exact differential first-body linear candidate for
nomination. It also shows that B-prime's failure was not explained solely by
learning its direction in the absolute final-position space.

The result does not prove that L13 lacks relation information: the twelve
paired deltas and cross-fold direction cosines show the opposite. It shows that
the information is not an absolute globally thresholdable scalar under the
tested representations. A paired contrast between two prompts remains a
diagnostic oracle, not a deployable single-prompt detector.

No second anchor, normalization, position atlas, nonlinear model, threshold,
or detector/effecter reinterpretation is attempted. Under the progressive
protocol, this arc nominates no learned architecture. The compiled
product-bounded branch remains the honest next implementation path unless a
future research arc introduces a structurally different observable and fresh
grouped material.

## Evidence boundary

- Analysis SHA-256:
  `f7303c026c2eba2ffecc97302b1d6ab45ba2f3aaf7c0fd678219389a790c938a`.
- Frozen manifest SHA-256:
  `4a2abc39e5777d135af43cf92aa2e377fd7809b199dbcfc782d229a4fc6bd380`.
- Position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`.
- No E4B process, detector artifact, threshold, payload, terminal adapter, ABI,
  calibration, holdout, or push was used.
