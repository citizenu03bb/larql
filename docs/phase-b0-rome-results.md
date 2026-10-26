# Phase B0: rank-one solver and model-free validation

B0 is complete on `research/phase-b-rome`, created from
`research/edit-certification` at `2fdc844f312f65b6608e79c310848123e5f9dcb8`.
The full phase-B decision brief was read. **No pretrained model was loaded
or forwarded; B1 and B2 were not launched. The blind directory was not
read or listed.** The old exploratory worktree and its unfinished L14
runner remain untouched. No push was made.

The method lives separately in
[`tools/rome_solver`](../tools/rome_solver/README.md), so no harness source
or certification rule changed. It accepts the existing instrument's model,
layers and tokenizer, captures mean raw down-projection keys, optimizes
one value vector with frozen weights and an essence KL, and exports the
existing `low_rank` format. It refuses blocks after 13 on E2B or 22 on E4B.
The update is the constrained rank-one form of
[Meng et al.'s ROME](https://rome.baulab.info/); the fixed Gemma adaptation
uses a shared absolute raw output value, not an answer-time residual edit.

Every edit saves float32 U0/V0 and the model-free algebra witnesses k*,
v*, **Wk*** and explicit C, plus context keys/values, complete seen-phrase
provenance, counts, optimizer trace and hashes. Wk* is necessary to
recompute U without loading W. The mean-key constraint and nullspace
preservation are tested on identity and nonidentity SPD C. The identity
ablation reuses the same optimized value and changes only C, with zero
new model calls.

Validation results:

| Suite | Outcome |
|---|---|
| New CPU synthetic method tests | 16 passed |
| Existing torch backend tests in the model venv | 6 passed |
| Existing harness pytest suite in system Python | 57 passed, 6 torch skips covered by the separate run |
| Fixed-setting synthetic rehearsal | 130 synthetic solve forwards, 25 Adam updates; separate 8-forward covariance rehearsal |

The tests cover correct layer/last subject token/raw pre-norm output,
unchanged non-target rows, no weight gradients or mutations, hook/state
cleanup on errors, float32 rank-one algebra and nullspace preservation,
byte-identical repeated solve/save, low_rank save/load versus dense update,
K/V deadlines, fixed corpus sampling and streaming covariance. The actual
registered optimizer settings are exercised by the standalone rehearsal.
Its maximum mean-key residual is 8.94e-8. Synthetic artifacts are explicitly
labelled and do not establish an edit of a fact or persistence.

Inspection without tensor-payload reads confirms the candidate projection
shape [1536, 6144]. Two architectural details are now recorded accurately:
blocks 15–34 use [1536, 12288], and the installed implementation applies
`layer_scalar` **after** the FFN and PLE residuals. The brief's earlier
scalar ordering is incorrect; optimization remains at the raw down_proj
output before `post_feedforward_layernorm`, as required.

The real local tokenizer was checked on all five public seen S1 requests,
all four context prefixes and the essence prompts, including the multi-token
target `swimming` and the final `aro` subtoken of Kilimanjaro. The covariance
sample is fixed now: 200×512 tokens, seed 20261026, English ADRs/emotion-train/
banking77-train quotas 40/80/80. Its hash and per-source/window witnesses are
committed. No real covariance activations were collected.

The [B1–B2 protocol](phase-b-b1-b2-protocol.md) fixes L4/L8/L12, 25 updates,
four prefixes, the KL/decay/clamp settings, covariance ridge, layer-selection
rule, controls, failure criteria and predictions. It uses full independent
dev certifications and a fixed identity ablation. The worst-case B1–B2
bound is 18,790 forwards and 375 Adam updates, each service still bounded
by 400/100, shared flock, 22G/24G/8G and 1800 seconds.

One interface is requested from Claude before B2: expose cap=400 through
public prepare/CLI, which still defaults to 24 at this base commit. The
[request](phase-b-harness-interface-request.md) does not patch the harness.
The actual phase-separated cost is four services per full dev fact, not
the decision brief's estimated three. This correction is in the protocol.

Proofs: [`phase-b0-method-tests.json`](research-data/phase-b0-method-tests.json),
[`architecture.json`](research-data/phase-b0-preflight/architecture.json),
[`dev-tokenization.json`](research-data/phase-b0-preflight/dev-tokenization.json),
[`covariance-sample.json`](research-data/phase-b0-preflight/covariance-sample.json),
[`budget.json`](research-data/phase-b0-preflight/budget.json), and
[`synthetic receipt`](research-data/phase-b0-synthetic/receipt.json).
The [artifact manifest](research-data/phase-b0-manifest.json) pins code and
evidence. Claude's independent algebra check remains to be done. Real NF4
autograd, float16 rounding, selectivity and reload persistence are untested
in B0 and await approved execution and independent certification.
