# Causal entity-relation activation patching: results

**Status:** closed with a valid exploratory negative result

**Evidence label:** exploratory; spent construction groups and one engineered
source patch only

**Model boundary:** exact Gemma 4 E4B native-Q4K vindex

## Result

Neither preregistered candidate passed discovery. L11 failed all four discovery
folds, and L13 failed all four discovery folds. The fixed selection rule
therefore selected no layer, the construction-confirmation gate had no selected
candidate to confirm, and no causal mediator was nominated.

The experiment is mechanically valid rather than infrastructure-inconclusive.
It completed the exact 240-forward budget, all 24 hook-disabled replays had
identical full-logit digests, all 192 patched forwards recorded exactly one
application and no rejection, all logits were finite, and maximum self-patch
drift was exactly `0.0` logits against the frozen `1e-4` tolerance.

The bounded conclusion is:

> On this exact E4B Q4K artifact, engineered L31 source patch, spent 24-prompt
> suite, and full-row final-position intervention, the results do not support
> L11 or L13 pre-FFN residual state as a construction-stable causal mediator of
> the entity-relation dependence.

This does not show that no narrower subspace, component, position, layer, model,
or unedited fact can mediate a relation. Those are different hypotheses. This
protocol does not authorize testing them as a rescue.

## Frozen discovery gate

For each candidate and fold, `m+` and `m-` are the signed forward and inverse
gap effects, `t+` and `t-` are the corresponding expected-token effects, `gap
spec.` and `target spec.` subtract the largest absolute entity-absent or cyclic
non-paired control, and `L12 adv.` is the matched-minimum advantage over the
same-fold L12 control. Every sign must be positive and every margin must be
strictly greater than `0.001` logits.

| Site | Fold | m+ | m- | t+ | t- | gap spec. | target spec. | L12 adv. | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| L11 | 0 | -0.135559 | 0.032454 | -0.093283 | 0.012636 | -1.077065 | -0.516165 | -0.012997 | no |
| L11 | 1 | -0.482006 | 2.567774 | 0.465952 | 2.408651 | -4.098274 | -0.175037 | -0.151722 | no |
| L11 | 2 | -3.928612 | 0.022268 | -3.609108 | 0.071459 | -9.104851 | -8.580868 | -1.882681 | no |
| L11 | 3 | 0.042336 | 0.584064 | 0.044743 | 0.024384 | -4.465862 | -4.443134 | -0.131480 | no |
| L13 | 0 | -0.024864 | -0.086447 | -0.160154 | -0.152754 | -2.257339 | -0.532921 | 0.036116 | no |
| L13 | 1 | -0.333117 | 2.107311 | 0.471672 | 1.973930 | -3.514107 | -0.146379 | -0.002832 | no |
| L13 | 2 | -2.163219 | 0.301769 | -1.857466 | 0.323782 | -3.830113 | -3.350613 | -0.117289 | no |
| L13 | 3 | 0.231272 | 0.698023 | 0.324095 | -0.325214 | -5.024242 | -5.570049 | 0.057456 | no |

The failure is not marginal. Each candidate has wrong-direction effects in
multiple discovery folds, and every discovery-fold causal-specificity margin
is negative. In other words, even the correctly signed matched effects do not
strictly exceed the frozen entity-absent and non-paired-construction controls.

## Controls and falsifier

L33 confirms that the intervention can strongly change output logits: its
matched minimum reaches `4.933346` logits on fold 3 and `2.220364` on fold 5.
Those effects are not relation-specific. The largest same-fold control gap is
`18.359726` on fold 3 and `2.302191` on fold 5, leaving negative specificities
of `-13.426380` and `-0.081827`.

The preregistered falsifier therefore wins. Whole-row replacement can produce
large output changes, but those changes are at least as compatible with donor
wording, construction, entity identity, position, or late edited-output state
as with a stable entity-relation mediator. The experiment does not distinguish
among those nuisance explanations and makes no stronger causal claim.

Folds 4-5 were executed as frozen spent construction-confirmation material, but
no candidate reached their selection gate. They cannot nominate a candidate
post hoc. Their per-fold candidate gates were also false, and L13 cannot replace
an L11 failure under the frozen rule.

## Execution validity

| Item | Frozen or observed value |
|---|---|
| Source commit | `befbe11f44708f32cb6fc9ad43d5d90238943dcc` |
| Base commit | `c628a7cc3044c52114410905d897f75af2be95d8` |
| Unit | `larql-causal-entity-relation-befbe11f.service` |
| Invocation ID | `ac9b73aefd384c079f660d4fb85b3663` |
| Wall time | `57min 7.178s` |
| CPU time | `9h 51min 38.082s` |
| Peak memory | `22G` |
| Peak swap | `4.1G` |
| Exit | `0/SUCCESS` |
| Baseline/donor forwards | 24 |
| Hook-disabled digest replays | 24/24 identical |
| Patched forwards | 192 |
| Total forwards | 240 |
| Maximum self-patch drift | `0.0` logits |

The first cold release build exhausted the root filesystem before any model
load. Its generated cache was removed, and the same committed source was built
successfully using the existing LARQL release cache. This was a pre-run
infrastructure event, not a model result. At launch and completion, the root
filesystem retained about `6.8G` free; the completed raw result is only 151,052
bytes.

## Integrity

| Artifact | SHA-256 |
|---|---|
| Frozen preregistration | `9f7b0ceb2a03ffa2e670798268b1d39e78ffa8d906a9ae5ced2bbafc2cd2eef4` |
| Runner source | `db4721ff2b45bcb183628fc2cc4d6efb6f55e5130cd5cf8910680d6162ec96b8` |
| Frozen analyzer | `5b0aad78cd67445b8a1cab1e68d8297f89444a40286cebd81512ffec034359b0` |
| Analyzer tests | `5013a2bb0e7f13d4afc8de4b0c07090e153e427bc11cd672fc85337a8fa76abb` |
| Release binary | `b0de923d6e9812116ef8f4d8d42780b9523f64225340d47690dbd19d21baedb0` |
| Spent suite | `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243` |
| Source patch | `2c69ccd022cb2c24d332eb1acf0f92e5c4a7fb12ec09bd21732de27bea8433fd` |
| Private pre-run manifest | `7f34f516e9b1aa2242b12d569ae232929d9f055749456458e01838ad36844f68` |
| Raw 240-forward result | `74925b7d01081847557cd2e7dc2355fd39833c47bc0c7ecce5b6d8b6ef61ad9c` |
| Frozen analysis result | `b436941a0044af9579d7d218d7dc94915c9214c4c5a63b9305cbed1ef5328a7f` |

The vindex binding remains
`sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
The private pre-run manifest contains the exact paths and hashes for the vindex
index, tokenizer, Q4K manifests, protocol, corpus, patch, binary, runner,
analyzer, and output destinations.

## Stop decision

This causal full-residual phase is closed as a valid negative. No candidate,
detector, patch artifact, ABI change, sublayer decomposition, added site,
alternative distractor, holdout access, or second heavy run is authorized by
this result. Any future causal mechanism study requires a separate hypothesis,
new checksummed protocol, and new authorization; it must not be presented as a
rescue of L11 or L13.
