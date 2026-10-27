# Codex P1 GPT-2 XL port — frozen before its first real solve

Pat's brief 40 authorizes these five solves under approved A1: BMW→Italy,
Germany→Finnish, Gretzky→baseball, Nintendo→France, Shakespeare→Spain.
Their exact S1 requests, leading-space targets and subjects are hardcoded
from A1 and checked against the qualified public specs before execution.
The existing 21 templates are hash-checked, never regenerated. The shared
C17 covariance is pinned once before all five solves. No peer solve source
or peer factors are read before P1-4.

## Published objective and exact choices

The authority is the published GPT-2 XL
[hyper-parameters](https://github.com/kmeng01/rome/blob/main/hparams/ROME/gpt2-xl.json),
[value optimization](https://github.com/kmeng01/rome/blob/main/rome/compute_v.py)
and [key direction](https://github.com/kmeng01/rome/blob/main/rome/compute_u.py).
They are read as reference only; no downloaded code is executed.

The port uses an additive delta at the subject's last subtoken of the
block-17 **MLP output**, across the 21 rewrites and one `{subject} is a`
KL prompt. Adam has lr 0.5, betas (0.9,0.999), eps 1e-8. The penalty is
`0.5 * ||delta|| / ||clean_native_value||²`; KL(edited || native), weight
0.0625, is measured at the KL prompt's subject position. Delta is clamped
to four times the clean native value norm. All model parameters are frozen.
The final block-47 representation, final norm and tied language head
define target likelihood. Selecting only loss-relevant head positions
avoids unnecessary vocabulary tensors without changing that objective.

**Iteration convention:** A1 specifies 20 gradient steps. This port makes
exactly **20 actual Adam updates**, with no loss-dependent early stop.
The reference source has an early-stop condition and breaks before the
last update of its nominal 20-iteration loop. This difference is disclosed
before execution, rather than hidden in a claim of bit-identical reference
execution. It does not change A1's stated hyper-parameters or sequence cap.

## Affine Conv1D and saved constraint

GPT-2's c_proj stores `[6400,1600]` weights, input by output, and a bias.
The mean of the 21 captured subject keys determines `q=C^-1 mean(keys)`;
the input factor is `V=q/||q||`. The clean first-template key is k*.
The clean served output is **Wk* + b**, including the bias. Optimization
sets v* to that clean output plus delta. The output factor is
`U=(v*-(Wk*+b))/(Vᵀk*)`. The physical weight update would be `V @ U.T`;
the existing low_rank backend applies the equivalent `(x @ V) @ U.T`
to every token, without modifying the baseline checkpoint.

This follows the published mean-direction/clean-context constraint
construction. It does not silently reuse E2B's shared-absolute-value
optimizer or mean-key constraint. Those source files remain frozen.
SPD solves use float64; factors and witnesses are float32; served model
activations are fp16. The saved float32 affine constraint must pass
rtol/atol 2e-5. k_mean and all 21 context keys/values are also retained.

Prelaunch review corrected a guard before any real solve: a clean-key dot
with a mean-derived direction can be negative. The constraint divides by
a finite nonzero signed dot; rejecting a valid negative dot would be an
implementation error. A synthetic counterexample and fresh full rehearsal
cover this correction. No model/data parameter is changed.

## Counts, rehearsals and outputs

Each fresh model service uses 21 native extraction sequences, one native
KL sequence and 20 batches of 22 sequences: **462 sequence-forwards and
20 Adam updates**, within the authorized 500/25 cap. There is no extra
generation, calibration, warmup or final validation forward. Every call
counts before execution. Each solve has a fresh tiny random GPT-2 CPU
rehearsal using all 21 shared templates and the same optimizer/count path.
It performs 462 synthetic sequences, 20 updates and checks the affine
constraint. Unit tests additionally cover multi-token alignment, immutable
model weights, Conv1D bias/orientation and low_rank save/load application.

Actual services use shared flock, bounded systemd (22G/24G/8G, 1800 s),
deterministic CUDA and offline loading, one solve per service. Failure,
nonfinite data, hash drift, budget or shape error stops the series with no
automatic retry. Every candidate starts from the unchanged base model.

Each `solves-codex/<fact>/` contains backend.json and U0/V0 plus k*, k_mean,
v*, Wk*+b (`wk_plus_bias.npy`, also `w_k_star.npy` with explicit affine
metadata), bias, delta, context witnesses, all training input/target IDs,
optimizer history, seen phrases and hashes. A complete status is written
last. Service progress/errors live separately in `solve-services-codex/`.
The artifacts are candidates for cross-check/certification, never a
self-issued certificate. No push or blind-set access is authorized.
