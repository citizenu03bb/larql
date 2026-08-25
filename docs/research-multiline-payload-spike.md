# Surface-specialized multi-line payload spike

**Status:** closed at the activation-margin gate on 2026-08-25. Cached
validation was not run.

## Question

Can a second FFN payload row, addressed from a passive surface form, lift a
first target token that the existing canonical row misses without changing the
frozen contextual detector or affecting rejected controls?

This is not an Alder-v2 experiment and cannot certify another fact. It uses the
already-spent Galadriel development material precisely so the payload mechanism
can be engineered without touching either sealed Alder holdout.

## Fixed inputs

- Exact base artifact:
  `gemma-4-e4b-upstream-20260811-q4k.vindex`.
- K=1 patch: `galadriel-compose-q4k-contextual.vlp`, L31/F1625.
- Address capture:
  `position-validation2-residuals-20260815.json`.
- Address prompt: `Lothlorien is governed by`.
- Evaluation suite:
  `multitoken_t3_n1_factorial_validation2.json`.
- Target IDs: `[8833, 553, 21457]`.
- Existing frozen L6 conjunction detector, copied unchanged to both rows.

The address prompt is deliberately reused for synthesis and measurement. Its
result is therefore only an in-sample mechanism check, never a generalization
estimate. The surrounding positive prompts and factorial controls are also
previously opened development material.

## K=2 construction

1. Apply the K=1 patch to the bound base overlay and ask
   `find_free_feature(31)` for a distinct second slot.
2. Read the pre-FFN L31 residual captured for the address prompt.
3. Set the new gate and up directions to the unit-normalized address residual.
4. Preserve the L2 norms of the K=1 gate and up rows.
5. Copy the K=1 down vector unchanged, so the experiment changes addressing
   capacity rather than target direction or payload scale.
6. Copy the same contextual detector and target IDs to the new slot.

A duplicate of the existing gate/up direction would be equivalent to a gain
increase and is outside this spike. An exploratory screen performed before
freezing this document found cosine 0.209 between the existing gate and the
passive address residual, so the candidate is structurally distinct enough to
run. That screen is diagnostic, not acceptance evidence.

## Gates

The spike advances only if all of the following hold:

1. The derived patch contains two distinct `Insert + Context` slot pairs,
   preserves the source row, target IDs, down vector and base checksum, and
   passes patch parsing/application tests.
2. On `Lothlorien is governed by`, the K=2 model selects target ID 8833 as
   top-1 with a logit margin of at least `+0.5` over the runner-up. The K=1
   capture selected ` the`, with target probability about 0.397.
3. The other two positive prompts in validation2 still select the first target
   ID and retain positive detector decisions.
4. Every one of the nine factorial controls receives only detector rejections;
   no rejected row may appear as an effective patch activation.
5. A cached K=2 validation emits the exact three-token span and releases on all
   three positives, while all nine controls remain rejected.

The K=2 activation capture runs first. Failure of gates 2 through 4 stops the
spike before cached validation.

## Resource rule

Every full E4B process runs sequentially in a unique bounded `systemd --user`
unit with `MemoryHigh=22G`, `MemoryMax=24G`, and `MemorySwapMax=8G`. Host
processes are checked before each launch. A cgroup kill ends the run; limits are
not relaxed.

## Interpretation

- Passing supports only the narrow claim that surface-specialized rows can add
  payload addressing capacity on spent development material.
- Failing rejects this particular single-extra-row construction. It does not
  license scale tuning, a third row, or work on Alder inside this spike.
- Alder-v2 still requires a separate preregistration, fresh captures, a new
  passive/agentless dev suite, unchanged LOSO acceptance, and one final holdout
  open only after both payload and detector gates pass.

## Result

The builder allocated a genuinely distinct second slot, L31/F7114. The derived
gate preserved the K=1 norm (32.7989), the derived up row preserved its norm
(1.0000), and the down vector was copied exactly. The source gate versus passive
address cosine was 0.2091.

On the three positive validation2 prompts, K=2 selected the first target ID in
3/3 cases. The two already-passing prompts retained their behavior:

| Prompt | K=1 target probability | K=2 target probability |
|---|---:|---:|
| `The monarch of Lothlorien is` | 0.393852 | 0.398068 |
| `Lothlorien's monarch is` | 0.452894 | 0.457728 |

For the address prompt, K=1 selected ` the` at 0.453812 while ` Gal` had
probability 0.396943, a target-versus-winner logit margin of -0.133890. K=2
flipped top-1 to ` Gal` at 0.441717 versus ` the` at 0.427188, but the resulting
margin was only +0.033443. The improvement was +0.167334 logits, far below the
predeclared absolute margin gate of +0.5.

All nine factorial controls were rejected by both copied detector instances,
and all nine exposed zero effective patch activations. Gates 1, 3, and 4
passed; gate 2 failed. Per the stop rule, cached exact-span validation was not
run and no scale, third row, or alternative address was tried.

The narrow conclusion is that a surface-specialized second row can flip the
first-token decision while preserving contextual rejection, but the
same-norm/same-down K=2 construction does not create a robust payload margin.
This points away from naive row count and toward a mechanism that can add
output-space capacity, such as a small conditional residual adapter. It does
not justify Alder-v2 yet.

The run used the bounded unit
`larql-e4b-k2-passive-activation-20260825-1300`: 3 minutes 31.9 seconds wall
time, 22 GiB cgroup memory peak, and 4.5 GiB swap peak.
