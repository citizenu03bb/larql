# A3 R1–R11 audit against Codex's executed port — 2026-10-27

This responds to Pat's brief 43 and reviews A3 at **21f9ce36**. It reads
Codex's unchanged source, archived arrays and traces, the local tokenizer,
and the upstream reference as text. It loads **no pretrained weights**,
executes **zero model forwards**, reads no peer solver implementation,
and grants no approval for A3. No source or candidate is repaired.

**The mathematical specification agrees with the port, with A3's explicit
R8 exception.** The revised budget arithmetic is correct. Before approval,
correct the reported loss range and qualify the statements of identical
execution: the native KL capture and numerical conventions differ too.
Those differences are not new evidence of an equation or algebra error.

## Point-by-point review

The source hashes in all five executed solver manifests match the current
Codex files. The original P1-4 audit replays unchanged. This table combines
source inspection with the recalculations below; it does not claim that
unarchived logits or gradients have been reconstructed without a model.

| Rule | Finding against the frozen port |
|---|---|
| R1 | Correct equation: unit direction of C⁻¹ times mean of 21 keys. Codex uses a float64 Cholesky solve, then stores float32 V, rather than the reference inverse/matmul arithmetic. Independent C·V direction and unit-norm checks pass. |
| R2 | Correct for all five frozen targets: 21 rewrite rows plus one KL row, right padding. All targets are one token, so the appended target prefix is empty. All 110 subject-last positions and saved token sequences match an independent transcription of the reference rule. |
| R3 | Correct additive zero-initialized delta on every row, KL included, at the MLP output. Codex explicitly casts delta to the output dtype before addition; the reference uses mixed-dtype in-place addition. These need not round identically. |
| R4 | Correct clean base: saved base equals the first captured native output exactly. Capture occurs during singleton key extraction, before optimization, rather than in the first 22-row optimizer batch. The base is promoted to float32. |
| R5 | Correct target-mean NLL and mean over 21 rows, from final block/ln_f/tied head. Codex promotes logits to float32 before cross-entropy. The logit precision is an implementation convention, not fixed just by the equation. |
| R6 | Correct KL direction, coefficient and subject-last position. **Codex captures native KL in a separate singleton forward before optimization**, rather than in the first optimizer iteration. Its log-softmax is float32. This qualifies A3's “same” entry. |
| R7 | Correct unsquared penalty. Delta and base norms are float32. Recalculating decay from the preceding post-update delta norm agrees with the stored, already weighted decay. |
| R8 | Exactly 20 effective Adam updates, lr 0.5, no early stop, clamp after every update: the explicit A3 exception. All archived losses exceed 0.05. Loss terms are pre-update; each row's delta_norm is post-update. |
| R9 | Correct clean affine constraint including bias. Saved target equals base + delta exactly. The denominator dot product uses float64 and U is saved in float32. All five independent affine checks still pass. |
| R10 | Correct approved adapter format: U=(1600,1), V=(6400,1), scale 1, c_proj Conv1D update V·Uᵀ. This is the declared low_rank adapter, not newly baked model weights. |
| R11 | Correct leading spaces in the five fixed requests, checked against the qualification specs before their execution. |

The machine-readable table provides precise source locations. No real
forward is needed for the tokenizer or the saved-array checks.

The official sources consulted are pinned at
`0874014cd9837e4365f3e6f3c71400ef11509e04`:
[value solve](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/compute_v.py),
[key solve](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/compute_u.py),
[subject lookup](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/repr_tools.py),
[request preparation](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/rome_main.py).
The code was read as text, never imported or executed.

## Correction to the loss claim in A3 §3

In Codex's manifests, **kl already includes 0.0625, and decay already
includes 0.5**. The archived total is nll + kl + decay.

| Fact | Last total loss | If kl were incorrectly weighted again |
|---|---:|---:|
| BMW | 0.147074260 | 0.114812112 |
| Germany | 0.180374801 | 0.172685544 |
| Gretzky | 0.132611886 | 0.122583850 |
| Nintendo | 0.225200999 | 0.187197778 |
| Shakespeare | 0.126851133 | 0.122321104 |

The correct final range is **0.126851133 to 0.225200999**, not 0.115 to
0.187. Reweighting the archived KL a second time reproduces the latter
rounded range. This is a diagnostic observation; Claude's recalculator
source has not been read. Every total at every archived iteration remains
above 0.05, so this reporting correction does **not** change the observed
absence of an early-stop crossing on Codex's trajectory.

That last statement concerns the archived trajectory. It is not a new
execution of the upstream implementation under different numerical
conventions, nor a measurement of the unlogged loss after update 20.

## Numerical execution needs an explicit qualification

A CPU tensor example with base=1 in float16 and delta=0.0004884 in float32
gives **1.0009765625** for mixed-dtype in-place addition, and **1.0** when
delta is cast to float16 before addition. Thus R3's two source expressions
are not universally identical in finite precision. This is a scalar
example, not a measured effect on the actual GPU solves or on Nintendo.

The same distinction between equations and execution applies to float64
moment solving, float32 log-softmax/regularization, and singleton versus
batched native captures. The tiny nonzero first-iteration weighted KL
values in the frozen traces are about 9.4e-8 to 3.0e-7; they are consistent
with a separately captured baseline and numerical differences. The
individual causes have not been isolated.

I recommend that A3 declare these Codex conventions and the corresponding
choices in Claude's independent rewrite **before any new real forward**.
An acceptable wording is “same mathematical reference, with numerical and
capture differences declared in advance, and the shared 20-update
exception,” rather than “the loop is the only execution difference.” Retain
the current candidates and thresholds; do not modify the frozen port or
change conventions after seeing a new comparison to force agreement.
The existing gate is a test of agreement, not proof that every disagreement
must be a mathematical implementation error.

## Revised budget and stop rule

The A3 table adds correctly:

- bench plans: 832 + 784 + 784 + 832 + 832 = **4064**;
- D1 probes: at most 5 × 54 = **270**;
- P1-5: at most **4334**;
- total since v2: 6720 + 2310 + 4334 = **13364**;
- including the separately reported 500 from v1: **13864**.

This audit checks the table's arithmetic; it does not independently rerun
plan.items. The actual plans must still be committed before P1-5, as A3
requires. There is no arithmetic objection to this corrected budget.

The 3-or-more-disagreements stop, no-A4 rule, unchanged thresholds and
native certification ceiling of 3/5 remain explicit. A3's broader approval
is pending; Pat's previously stated decision on the 20 updates does not
authorize new model services here.

## Deliverables

- `tools/rome_solver/audit_p1_a3_spec_codex.py`: reproducible model-free audit;
- [JSON evidence](research-data/phase-b-p1-a3-spec-audit-codex.json): source hashes,
  all eleven findings, tokenizer checks, losses, numerical witness and budget;
- two regression tests for already weighted traces and the mixed-precision
  example, alongside the existing P1 tests.

All **13 targeted P1 tests pass**. The largest relative C·V direction
residual across the five edits is 8.52e-7, within the declared float32
comparison tolerance.

All original P1-4 delivery files, the ten candidates and Codex's executed
solver files remain unchanged. No model service or push was performed.
