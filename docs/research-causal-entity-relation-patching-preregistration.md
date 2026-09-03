# Causal entity-relation activation patching: exploratory preregistration

**Status:** frozen 2026-09-03 after human approval and before the first model run

**Evidence label:** exploratory; spent construction groups only

**Model boundary:** one exact Gemma 4 E4B native-Q4K vindex

## Question and maximum claim

Does a full residual state at the final prompt position, immediately after
attention and before the FFN at L11 or L13, causally mediate the relation
dependence of the already-compiled `Archive Zephyr -> Nora Veldt` behavior?

The experiment can nominate at most one model-bound causal mediator for a
future fresh replication. It cannot certify a detector, establish a global
threshold, reopen a sealed holdout, or establish semantic understanding,
natural factual recall, or cross-model generality.

The source patch is necessary because `Archive Zephyr -> Nora Veldt` is a
fictional association and the unedited model has no expected Nora answer. This
also limits the claim: a pass would show causal mediation into this engineered
L31 payload, not that the same state mediates an unedited fact. A result that
only transfers late edited-output state is not evidence for the hypothesis.

## Frozen artifact and spent material

- Base vindex:
  `/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`.
- Base binding:
  `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
- Source patch:
  `/home/pat/Projects/larql-research/multitoken-span-emitter-v2/factorized-relation-v1/factorized-relation-v1-source-20260827.vlp`.
- Source patch SHA-256:
  `2c69ccd022cb2c24d332eb1acf0f92e5c4a7fb12ec09bd21732de27bea8433fd`.
- Spent suite:
  [`research-data/factorized-relation-v1/development.json`](research-data/factorized-relation-v1/development.json).
- Suite SHA-256:
  `ab0bc547f57bdfd0b8c230adba468df2df8832f45215679d05f9b115d0849243`.
- Rows: 24, comprising six construction folds with exactly one each of cells
  `11`, `10`, `01`, and `00`.
- Expected first answer token: ID `107545`, surface ` Nora`.
- Fixed distractor token: ID `506`, surface ` the`. It is frozen because it is
  the observed top competitor on both spent payload-miss positives and is
  common among their matched controls.

No calibration suite, factorized-relation holdout, Alder holdout, Alder-v2
holdout, product Minas Tirith material, or new prompt may be read by the
experiment harness.

## Inventory of reusable captures

The following captures are already spent and may be used only for site
justification, token-position verification, and integrity comparison:

| Capture | Rows | Stored tensor and positions | SHA-256 |
|---|---:|---|---|
| `factorized-relation-v1/development-capture-20260827.json` | 24 | L13 final-position raw pre-FFN residual only | `4b0a57229b8507cb0b15f83ef4c66539141d8e85fb9170e2daf9d7e3ed2822e3` |
| `internal-reference-b-prime/position-capture-20260827.json` | 24 | L13 raw pre-FFN residual at every encoded prompt position | `65981056487687ed7be74632c21529e5cfac6bf3ab1d8769cfc9439880548bf2` |
| `late-decision-relation-atlas/capture-20260828.json` | 24 | L29, L31, and L33 raw pre-FFN residuals at every encoded prompt position | `794e8361640b656c2402de7b2109bb549274bad29a4c0959e24df6cc0c93e5d3` |
| `full-depth-relation-atlas/capture-20260829.json` | 24 | L0-L41 raw pre-FFN residuals at every encoded prompt position, hidden size 2560 | `d4f9170e17e3ec6c13725776ffc2dda761861d7fdd601b9d46a891d5ed4036f1` |

All 24 full-depth rows have `position_residuals` row counts exactly equal to
their stored `prompt_ids` lengths. Prompt lengths range from 6 to 13 IDs,
including BOS. Therefore `last` is defined independently for donor and
recipient as `len(prompt_ids) - 1`; it is not a shared absolute index. The
existing `post-layer` patch helper, which patches the same absolute position in
both prompts, is not the instrument for this protocol.

## Frozen intervention

Each donor is forward-passed once in the same resident process as the patched
recipients. The runner records the complete 2560-dimensional residual row at
the donor's final encoded prompt position at all four frozen sites. A patched
recipient receives that row at its own final encoded prompt position, at the
same layer and exactly at the `post-attention / pre-FFN` hook point. Every other
position, layer, and component remains the recipient computation.

The intervention is full-residual replacement. There is no interpolation,
normalization, centering, subspace projection, attention/FFN decomposition,
head selection, token alignment, or gain. A sublayer decomposition is allowed
only under a separately checksummed protocol after a positive result here.

The four and only four site configurations are:

| Site | Role | Frozen reason |
|---|---|---|
| L11 / final / pre-FFN | candidate 1 | only isolated positive layer in the spent absolute atlas; earliest fixed priority |
| L13 / final / pre-FFN | candidate 2 | strong positive within-construction relation deltas despite failed global separation |
| L12 / final / pre-FFN | adjacent negative control | immediate neighbor of both candidates and failed the spent global-gap gate |
| L33 / final / pre-FFN | late sensitivity control | fixed late site; cannot be nominated or rescue an early failure |

No other layer or position may be added after results are inspected. If both
candidates pass discovery, L11 is nominated first because the fixed selection
rule prefers the earliest causal site. L13 is considered only if L11 fails the
discovery gate; it cannot replace an L11 replication failure.

## Frozen intervention arms

For each construction fold `c`, let `P_c`, `E_c`, `R_c`, and `N_c` denote
cells `11`, `10`, `01`, and `00`, respectively. Let `next(c) = (c + 1) mod 6`.
At every site, run exactly these eight patched arms:

1. matched forward: `P_c -> E_c`;
2. matched inverse: `E_c -> P_c`;
3. positive self-patch: `P_c -> P_c`;
4. control self-patch: `E_c -> E_c`;
5. entity-absent forward control: `R_c -> N_c`;
6. entity-absent inverse control: `N_c -> R_c`;
7. non-paired-construction forward control: `P_next(c) -> E_c`;
8. non-paired-construction inverse control: `E_next(c) -> P_c`.

The cyclic non-paired donor is fixed and never chosen by similarity or result.
It measures how much full-row transfer imports construction-specific wording
rather than the matched relation change.

## Outcomes and causal metric

For a prompt output, define the raw-logit gap

```text
g = logit(token 107545, " Nora") - logit(token 506, " the")
```

For an arm `A -> B`, define

```text
delta_g(A -> B) = g(B patched from A) - g(B baseline)
delta_t(A -> B) = logit_Nora(B patched from A) - logit_Nora(B baseline)
delta_d(A -> B) = logit_the(B patched from A) - logit_the(B baseline)
```

For each fold and site, the signed matched effects are

```text
m_plus  =  delta_g(P_c -> E_c)       # predicted > 0
m_minus = -delta_g(E_c -> P_c)       # predicted > 0
t_plus  =  delta_t(P_c -> E_c)       # predicted > 0
t_minus = -delta_t(E_c -> P_c)       # predicted > 0
```

The two preregistered causal-specificity margins are

```text
gap_specificity = min(m_plus, m_minus)
                - max(abs(delta_g(R_c -> N_c)),
                      abs(delta_g(N_c -> R_c)),
                      abs(delta_g(P_next(c) -> E_c)),
                      abs(delta_g(E_next(c) -> P_c)))

target_specificity = min(t_plus, t_minus)
                   - max(abs(delta_t(R_c -> N_c)),
                         abs(delta_t(N_c -> R_c)),
                         abs(delta_t(P_next(c) -> E_c)),
                         abs(delta_t(E_next(c) -> P_c)))
```

These paired causal margins, not classification accuracy, AUC, mean effect, or
top-1 rate, are primary. All component logits and distractor effects are
reported so an apparent gap improvement caused only by suppressing ` the`
cannot pass.

## Mechanical and scientific gates

The hook's model-free gate passes only if its deterministic unit test proves:

- a wrong layer is byte-for-byte unchanged;
- at the requested layer, the requested row exactly equals the donor row;
- every non-requested row remains byte-for-byte equal to its input;
- the hook does not write again at `post-layer`;
- an invalid position or hidden dimension is rejected and leaves the tensor
  unchanged.

During the E4B run, every patched forward must record exactly one applied site
and zero rejected sites. Every self-patch must satisfy

```text
max(abs(delta_g), abs(delta_t), abs(delta_d)) <= 1e-4 logits
```

The explicit hook-disabled replay of every prompt must have a full raw-logit
f32 digest identical to its baseline. Any failure is
`infrastructure-inconclusive`, not a scientific result.

Discovery uses folds 0-3 only. A candidate passes a fold only if:

- `m_plus > 0`, `m_minus > 0`, `t_plus > 0`, and `t_minus > 0`;
- `gap_specificity > 0.001` logits;
- `target_specificity > 0.001` logits; and
- its `min(m_plus, m_minus)` exceeds the corresponding L12 adjacent-control
  value by more than `0.001` logits.

The `0.001` floor is ten times the maximum permitted self-patch drift. A
candidate passes discovery only if every one of folds 0-3 passes. If neither
candidate passes, stop with no nomination. If both pass, select L11; otherwise
select the sole passing candidate.

Folds 4-5 are then read as a construction-confirmation subset. They are held
out from the selection calculation but are already-spent, previously inspected
material and are not called a fresh holdout. The selected candidate must pass
the identical per-fold gate on both folds. If it fails, stop; L13 may not
replace a failed selected L11. L12 and L33 are always reported but can never be
nominated.

## Falsifier

The central falsifier is that a whole-row transplant merely imports donor
wording, position, entity identity, or late edited-output state. That
alternative wins if either matched direction has the wrong sign, if the Nora
logit itself lacks the predicted sign, if matched effects do not exceed both
entity-absent and cyclic non-paired controls, if the adjacent L12 control is as
strong, or if the selected site fails either construction-confirmation fold.

A large L33 effect alongside failed L11/L13 gates demonstrates instrument
sensitivity but is a negative semantic-mediation result. A correct effect on
only `P -> E`, an average positive effect with one reversed fold, or a gap move
caused solely by the distractor is also a failure.

## Exact execution budget

One heavy resident E4B process is permitted after approval:

- 24 baseline/donor forwards, capturing all four sites in each pass;
- 24 hook-disabled repeat forwards for full-logit digest identity;
- `4 sites x 6 folds x 8 arms = 192` patched forwards;
- total: exactly 240 forwards, four site configurations, one model load, one
  analyzer execution.

The runner may cache donor rows and baseline logits inside the process. It may
persist target/distractor logits, top-k diagnostics, hook counters, and full
logit digests, but not use any diagnostic to alter execution. There is no retry
for a scientific failure, no extra site, no alternative distractor, and no
candidate substitution after construction confirmation.

Before execution, the protocol, suite, source patch, runner source, analyzer,
tests, release binary, vindex manifests, tokenizer, base binding, unit name,
and output paths are checksummed into a private pre-run manifest. The protocol
and code must be committed before the release binary is built and hashed.

The unit runs sequentially through `systemd --user` with:

```text
MemoryHigh=22G
MemoryMax=24G
MemorySwapMax=8G
OOMPolicy=stop
RuntimeMaxSec=7200
```

The real process list is checked before launch. A cgroup kill, timeout, network,
DNS, sandbox, filesystem, hook-accounting, digest, or non-finite-logit failure
is infrastructure-inconclusive and does not authorize a scientific retry.

## Stop rule and next authorization

Stop immediately if the model-free hook gate fails, either causal direction
fails, matched effects do not strictly clear the frozen controls and noise
floor, no candidate clears all four discovery folds, the selected site fails a
construction-confirmation fold, or the four-site budget is exhausted.

A pass nominates at most one tuple
`(pre-FFN full residual, selected layer, donor:last -> recipient:last)` for a
separate future replication using fresh material. It does not authorize a
threshold, patch artifact, detector, ABI extension, component decomposition,
holdout access, or another heavy run.
