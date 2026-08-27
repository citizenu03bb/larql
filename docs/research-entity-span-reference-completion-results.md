# Entity-span internal-reference arm: result

**Verdict:** scientific failure of the completed fourth reference arm

**Frozen source:** `4d9f4b87eba412f26f21f40e24fdc7e3440de96a`

**Evidence label:** exploratory mechanical completion

**Holdout or calibration accessed:** no

## Result

The total four-alias implementation executed once on the exact spent
all-position capture. Final-position parity remained exact for all 24 rows.
The estimator was unchanged: each fold learned `raw_conjunction` on `x_last`
from the other five construction groups and scored
`x_last - x_entity_span`.

The completed arm failed its frozen strict-separation criterion:

- global relation gap: `-0.995696`;
- eligible `11` versus `10` gap: `-0.735966`;
- minimum paired present-minus-absent delta: `+1.047018`;
- maximum absolute relation-present-minus-absent projection carried by the
  entity reference itself: `1.599299`.

All twelve paired score deltas remained positive, but construction offsets
still overlapped globally. More importantly, the entity-span reference was not
relation-neutral. In constructions where relation words precede the entity,
the causal L13 entity positions already contain information about that prefix.
Subtracting the contextualized entity can therefore remove relation signal as
well as construction offset.

## Consequence

All four predeclared B-prime references now fail under the original direction
learned on final-position residuals. The reference family is closed for that
estimator. No fifth reference, threshold, normalization, or layer is tried.

This result does not test a direction learned directly in a differential
space. Such a fit is an estimator change created after observing the earlier
failures, not a completion of B-prime. Under the progressive protocol it may
be evaluated only as a separately named exploratory architecture and would
require fresh grouped material for any evidential claim.

## Evidence boundary

- Analysis SHA-256:
  `0abcdb05cca637aa9f12873c21c9423c0f85c0ac3b710af538fb0d0006dff795`.
- Frozen completion manifest SHA-256:
  `b13fd7078fe29e8d36cdb4d28ce061980b4896b676caa1b89f5f6c462e920b5b`.
- Position capture SHA-256:
  `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2`.
- No model process, threshold fit, detector artifact, payload, terminal adapter,
  ABI, calibration, or holdout was used.
