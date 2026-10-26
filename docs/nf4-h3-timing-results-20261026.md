# NF4 H3 timing test — 26 October 2026

**Registered reading: partial/local early-site compatibility, not the complete
ordered pattern.** At L8 the number margin moves by **+9.6445**, versus
**−0.0039 at L14** and **+0.0469 at L21**, at the same absolute perturbation
norm. The L8 random direction moves it by +0.9648. The symbol effect is still
present at L8 (+14.6250), and weaker than at L21 (+19.7344).

The frozen monotone prediction L8 > L14 > L21 is not met. These observations
show a local depth-dependent susceptibility; they do not locate a unique
number read-out or prove mediation through K/V.

## Context and frozen protocol

The operator authorizes H3 after the [PLE factorial](nf4-ple-factorial-results-20261026.md).
That test does not support a sizeable number effect from the tested PLE swap.
The earlier operational H→He transfer gate remains failed. A different agent's
artifact-only PLE cross-check (`7c0abada`) agrees with the recorded margins and
readings; it is not a new model run.

The separate H1 study has an **undetermined registered outcome** (`c0b3c373`).
Its eight-by-eight donor/symbol matrix (`4e966c0c`) is **post hoc**, with no new
forwards. Its diagonal pattern weakens a strong spelling-only interpretation,
without becoming a registered H1 result. The H1 prose abbreviates the answer
prefix; the shared source helper actually reuses the saved token ids. This
test likewise uses the exact verified ids, not that prose abbreviation.

R and PLE having effects on symbol is not proof that either transports a
complete identity, or that the two paths are redundant. The PLE factorial
already shows a strong non-additive interaction.

Protocol, scheduler and model-free checker are committed at **`30d573de`**
before any H3 forward. Fixed post-block sites are **L8, L14 and L21**, zero
based, with one intervention at entity row 5 per forward. There is no cumulative
transplant, new question, donor, layer search or gain search.

The primary native prefixes are the previously generated symbol prefix through
`Hydrogen's chemical symbol is $\text{` and the number prefix through
`Hydrogen's atomic number is`. Readouts are bare H/He and ` one`/` two`.
They are resumed value boundaries, not integrated edited responses from empty.

## Common norm and controls

Native and full helium-subject forwards capture the entity row at all three
sites. The input donor differs only at prompt position 5. At L8/L14 the
donor-minus-native entity vector is normalized once to the stored L21 residual's
float16 norm, **N = 63.8967**. At L21 the stored vector is reused exactly.
The same layer-specific vectors are used on symbol and number. Their causal
entity-row native/donor captures agree bit for bit across the two questions.
All PLE inputs stay native.

The archived Gaussian direction is reused at every site and on both attributes,
with the same cast norm. This is one fixed direction, not independent random
replicates per site. Requested norms must agree within 0.1%, and actual rounded
row-change norms within 1% (observed maximum actual mismatch: 0.0034%). The same vector and layer support are checked from
the saved arrays; no amplitude is chosen after observing an effect.

Core budget is exactly 24 forwards:

- native and complete donor for both attributes: 4;
- R and random at three sites for both attributes: 12;
- zero at all three symbol sites and at L21 on number: 4;
- native and L8-R on two supplied-prefix off-targets: 4.

Native, full-donor and L21 R/random full logits replay the prior captures bit
for bit. All four zero controls are bit-identical to native. Each real addition
is also checked exactly, with all other rows unchanged. Lithium/carbon native
controls are freshly replayed; their R8 controls keep Li/C as top-1. This is
two local controls, not full-chat selectivity.

## Results

E is target-minus-native margin after R minus the native margin; X is the same
increment for the random control. Each row is one case.

| Site | Symbol E, He−H | Symbol X | Number E, two−one | Number X |
|---|---:|---:|---:|---:|
| L8 | **+14.6250** | +0.3984 | **+9.6445** | +0.9648 |
| L14 | **+2.4375** | +0.2559 | **−0.0039** | −0.4570 |
| L21 | **+19.7344** | +1.0566 | **+0.0469** | −0.0234 |

Native margins are −21.2656 for symbol and −18.0430 for number. The L8 number
margin reaches −8.3984, still below the native token; all R and random primary
top-1 predictions remain H/one. Full donors still select He/two.

An early site passes the frozen specific-number criterion if E_number is at
least +1, exceeds its L21 effect by at least +1, and exceeds twice the positive
part of its random effect. It matches the relative-timing pattern only if the
symbol effect does not exceed its L21 effect by more than +1. The complete
ordered category further requires both early sites to match and number effects
to decline by at least one logit at each step L8→L14→L21.

- **L8 passes the local pattern:** number improves over L21 by +9.5977 while
  the symbol effect falls by 5.1094. Its number effect exceeds its random control.
- **L14 fails the number criterion:** its number effect is slightly below L21.
- The **complete ordered category is false**. The predefined partial/local
  category applies. No alternative ordering or best-layer search replaces it.

The first token also stays native on both off-targets. The conditional release
phase has no eligible cases and is not launched: **zero continuation forwards**.

## Absolute energy is not relative dose

| Site | Actual row-change norm / native entity-row norm |
|---|---:|
| L8 | 0.956 |
| L14 | 0.468 |
| L21 | 0.615 |

These ratios are essentially identical for the two primary questions. Matching
absolute L2 energy does not match relative dose, layer gain, vector orientation
or nonlinear sensitivity. This limits a pure timing interpretation. Earlier
pulses also alter later residual history and K/V together, so this experiment
cannot uniquely assign the number effect to K/V.

## Execution and evidence

- **24 new forwards**, one service, 41.1 s, exit zero. Planned maximum was 48
  across core and conditional releases, with a cap of 24 per service.
- The shared `.heavy-model.lock` is acquired with `flock -w 3600` around
  `systemd-run --user --wait`, as required across NF4/LARQL worktrees. The lock
  snapshot is taken after completion, when no holder remains; it is not an
  observation of the lock while active. The launch command records the wrapper.
- Limits remain 22G high, 24G maximum, 8G swap maximum and 30 minutes. The final
  service log reports 7.9G peak memory and no swap.
- A synthetic dry run exercises all 24 scheduled core calls, eight changed
  branches, normal releases and the 24-call incomplete release cap. Hook
  support and zero applications are checked. These fixtures are not model
  evidence and are guarded from use as real source data.
- The model-free checker returns **`verified_complete`**, recomputing vector
  construction, cast/applied norms, exact support, untouched PLE, hashes,
  source replays, effects, all reading categories and release eligibility.
- The same coding agent writes this runner and checker. The independent PLE
  cross-check is prior context, **not independent verification of H3**.

Sources: [`run_nf4_h3_timing.py`](../scripts/run_nf4_h3_timing.py),
[`check_nf4_h3_timing.py`](../scripts/check_nf4_h3_timing.py), and
[compact evidence](research-data/nf4-h3-timing-20261026.json). Raw data is local
under `nf4-h3-timing-20261026/core`. A ZIP with source evidence, tokenizer,
frozen code and SHA-256 manifest is restored and checked without model weights.
Full model weight shards are not hashed in full. No weights, vindex or
persistent patch change; no push or publication occurs.

## Next decision, not run

A clean next question is whether the L8 number effect has already left the
entity row by L14. Capture the full post-L14 state produced by that exact L8
pulse, then compare full-state replay, entity-only transfer and complementary
non-entity transfer, with native/sham/random controls. A full-state
reconstruction control must pass before interpreting either partition.
This holds the successful L8 intervention fixed instead of searching gains or
more layers. It still requires a new frozen protocol and budget. No such
partition or cumulative transplant is added to this series.
