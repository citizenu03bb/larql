# Natural Binding Causal Pilot v1 — model-free design

Status: **proposed protocol for review; no model forwards executed**. 2026-09-24.
Continuation of the task `Review semantic causal patching`
(`01a0660b-9f18-7ac1-8808-4ef824b136b7`). Technical base: `1b900054775b1c771bb87644e9c3cdad997f4895`.
Isolated branch: `research/natural-binding-causal-pilot-v1`.

## Decision in one page

**Question.** On the unedited Gemma 4 E4B Q4K artifact, can one rank-one
entity-by-relation interaction direction at a fixed middle layer transfer the
answer preference between two naturally learned relations, with effects that
survive a wording change and exceed entity-absent, wrong-entity, and
norm-matched perturbation controls?

**Single proposed intervention.** At zero-based L21, final prompt position,
post-attention/pre-FFN, replace only the donor's scalar projection onto a unit
direction `u`: `h' = h + u * dot(u, donor - h)`. Fit `u` once from the leading
right singular vector of four discovery interaction contrasts. Rank is one;
gain is one; there is one site and no candidate search. L21 is the architecture
midpoint `floor(42/2)`, a deliberately narrow sentinel choice, not a layer
nominated by previous measurements. Failure says nothing about other sites.

**Material.** Eight element names; chemical symbol and atomic number as the two
relations. Discovery uses hydrogen/helium and lithium/beryllium, with two
constructions. Confirmation uses boron/carbon and nitrogen/oxygen, with two
different constructions. There are 16 factual prompts and four entity-absent
prompts per stage. New corpus text and split membership are authored before
any inference. Confirmation is sealed against inference and tuning, not hidden
from the protocol author. Old LARQL holdouts remain untouched.

**Budget.** Per stage: 20 baseline/capture forwards + 20 hook-disabled replays
+ 16 recipients × 9 intervention arms = **184 forwards**. Maximum **368**:
184 discovery, then at most 184 confirmation. Early stops spend less; no unused
budget may be redirected to new configurations or extra attempts. Every
factual prompt is a recipient in both relation-transfer directions through
the ordered donor mapping. Captures and logits come from the same baseline
forward, with no additional fitting forwards or generation.

**Gate.** Baseline must naturally rank the complete correct answer token top-1
on all 16 factual prompts in that stage. Then, for every factual recipient,
both matched and cross-construction transfer must increase the alternate
answer's logit and its gap against the original answer by **more than 0.1**
logits. For each metric, the smaller of those two effects must exceed the
largest absolute negative-control effect by **more than 0.001** logits. Require
all 16 recipients, both relations and both constructions; never average away
a failure. Self-patches must drift by at most `1e-4` over the entire logit vector and in each of the two metrics.
Every disabled-hook replay must have the same full-logit digest as baseline.

**Falsifier and stop.** Incorrect baseline knowledge is an eligibility stop,
not a causal null. Mechanical failures are invalid experiments. Otherwise,
one failed causal row closes this configuration as a valid negative; do not
open confirmation after discovery failure. A confirmation failure closes the
pilot. No alternative rank, site, gain, element, prompt, target, threshold or
metric after model results. Freeze `u` and the discovery verdict before any
confirmation inference; confirmation must not refit `u`.

**Maximum positive claim.** One fixed subspace mediates answer preference for
these two relation classes, with entity-specific transfer on four new elements
and two new constructions, on this exact unedited artifact. This is neither a
general semantic-binding result nor a detector/edit compiler or proof of
understanding. Different answer types (symbols versus numbers), word identity,
and the small single-domain sample limit interpretation even on a full pass.

## Exact material and references

The fact table is transcribed from the
[IUPAC periodic table, 4 May 2022](https://iupac.org/wp-content/uploads/2022/07/IUPAC_Periodic_Table-04May22_CRA.pdf),
checked on 2026-09-24. Prior model knowledge is **not** assumed from the source;
it is tested by the eligibility gate.

| Split | Pair | Element | Symbol | Atomic number |
|---|---|---|---|---:|
| discovery | 0 | hydrogen | H | 1 |
| discovery | 0 | helium | He | 2 |
| discovery | 1 | lithium | Li | 3 |
| discovery | 1 | beryllium | Be | 4 |
| confirmation | 2 | boron | B | 5 |
| confirmation | 2 | carbon | C | 6 |
| confirmation | 3 | nitrogen | N | 7 |
| confirmation | 3 | oxygen | O | 8 |

Constructions, including the literal trailing newline:

```text
D0: The {relation} of {entity} is:\n
D1: {entity} has {relation}:\n
C0: For {entity}, the {relation} is:\n
C1: Regarding {entity}, give its {relation}:\n
```

Relations are exactly `chemical symbol` and `atomic number`. Entity-absent
prompts substitute `this element`. Their baseline outputs have no correctness
gate. They are donor controls, not inputs for direction fitting.

Tokenization preflight requires the prompt encoding to be an exact prefix of
the encoding of prompt plus answer, with exactly one added token for either
answer. All factual answers must round-trip exactly, and answer IDs must be
distinct across the 16 answers. Use the vindex tokenizer and architecture-aware
`encode_vindex_prompt` in the actual runner; explicitly verify one BOS token.
There is no chat wrapper, forced whitespace token, answer alias, teacher-forced
completion, or decoding step. Record exact IDs, lengths and final positions.
Any discrepancy with the Rust runner is a preflight stop before model loading.

The selected entity names had no matches in the prior tracked
`docs/research-*-results.md` reports or the opened
`factorized-relation-v1/development.json` suite. This is a bounded freshness
audit, not proof against every historical transcript. The corpus is newly
authored and has not been run here; old sealed corpora were not inspected.

## Fitting and intervention definitions

For each discovery pair `(a,b)` and construction `t`, construct one row:

```text
X[pair,t] = h(a,symbol,t) - h(a,number,t)
          - h(b,symbol,t) + h(b,number,t).
```

This cancels exactly additive entity, relation and construction contributions;
nonlinear lexical interactions can remain. Stack four rows, run float64 SVD,
and take the first right singular vector. Do not center, whiten, normalize
each row, fit to logits, or tune rank. Canonicalize sign so its largest absolute
coordinate is positive, with the first index breaking ties. Abort if `s0 <=
1e-8` or `s0-s1 <= 1e-8*s0`, or if any value is nonfinite. These are numerical
degeneracy checks, not scientific significance thresholds. Projection is
computed in float64 and cast to f32 once for the hook; store the exact final
f32 replacement row used at each intervention.

For each recipient `(e,r,t)`, the nine arms are:

| Arm | Donor or perturbation |
|---|---|
| matched | same entity, opposite relation, same construction |
| cross_construction | same entity, opposite relation, other construction in that split |
| wrong_entity | other entity in its fixed pair, opposite relation, same construction |
| entity_absent | `this element`, opposite relation, same construction |
| self | recipient itself |
| random_matched_plus/minus | ± a fixed direction orthogonal to `u`, norm equal to matched perturbation |
| random_cross_plus/minus | same ± direction, norm equal to cross-construction perturbation |

Each random direction is deterministic per recipient. Coordinate `i` is
`uint64_be(SHA256("NBCP-v1|<row-id>|<i>")[0:8])/2^64 - 0.5`;
remove its projection on `u`, normalize, and abort without reseeding if the
remaining norm is at most `1e-12`. Both signs are run, and both belong in the
largest-absolute-control denominator. This controls perturbation magnitude
and a fixed unrelated direction; it is not a statistical sample of all
directions. Unlike a wrong-entity donor, the cross-construction donor is a
required positive transfer arm, not a nuisance to suppress.

The existing `PreFfnPatchHook` can apply a precomputed replacement row;
it must record exactly one application at `(21, final_position)` and no
rejections. It must leave every other row/layer and the post-layer hook alone.
No `.vlp`, L31 payload, overlay, `ActiveSpan`, or changed base weight is used.
Use the unpatched exhaustive backend from the prior research harness, including
the native-Q4K path and model-configured final logit softcap of 30. Compare
post-softcap logits consistently. Verify implementation parity before running.

For recipient with ordinary answer `a` and alternate-relation answer `b`:

```text
gap = logit(b) - logit(a)
delta_gap = gap_after - gap_baseline
delta_target = logit_after(b) - logit_baseline(b)
```

Both ordered transfers must be positive relative to their own donor answer;
there is no post-hoc sign flip. The negative controls are wrong entity,
entity absent, and all four random arms. Apply the strict row gate separately
to both metrics. Record every row, including failed ones. No pooled p-value
or mechanistic generalization is claimed from this small pilot.

## Execution contract and approval boundary

The inherited handoff explicitly says:
> Ne charge aucun modèle avant mon accord sur ce document.

This package is the model-free document for that review, not authorization
already granted for 368 forwards. The current user request identifies the
research continuation and authorizes this preparation.

Expected artifact is
`/mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex`, with recorded
binding `sha256-manifest:c53cc460e0809529289375db6a90d84d038586d7cc005f19e401c8264066987a`.
Here “unedited” means this extracted, instruction-tuned checkpoint with no
LARQL overlay; it does not mean a pretrained non-instruction-tuned checkpoint.
The current index metadata confirms 42 layers, hidden size 2560, native Q4K,
and final logit softcap 30. Layer bands in the index are not scientific evidence.
Full weight integrity has not been rehashed in this model-free phase.

Before any approved inference, implementation must:

1. Build the Rust runner and run model-free integration tests proving the actual
   hook changes only the selected coordinate, rejects malformed/nonfinite data,
   records exactly one application, and matches the numerical reference after
   the f32 cast. The Python checks do not establish Rust hook integration.
2. Implement the strict whole-stage analyzer, eligibility checks, full-logit
   drift/digest gates and discovery-to-confirmation gate. The supplied Python
   module specifies the row gate and budget; it is not an inference runner.
3. Freeze checksums of the reviewed protocol, corpus, ledger, runner, analyzer,
   tests, compiled binary, tokenizer and vindex manifests in a new run manifest;
   bind it to a committed revision. Model-free hashes in this package are a
   design snapshot, not the future execution freeze.
4. Use separate discovery and gated confirmation commands. Validate metadata
   and tokenizer parity before either loads weights. Save the discovery verdict
   and fitted direction hash; require them at confirmation startup. Missing,
   failed or mismatching gates refuse confirmation before weight loading.
5. Persist an append-only forward ledger, including attempted/failed forwards.
   A stage-local crash cannot reset the budget or silently replay completed jobs.
   Stop on infrastructure failure; report it separately and obtain a revised
   execution plan rather than automatically retrying.
6. Check actual processes, then run sequential bounded `systemd --user` units
   with `MemoryHigh=22G`, `MemoryMax=24G`, `MemorySwapMax=8G`, unique unit names,
   and `RuntimeMaxSec=90min` per stage. Never remove limits after an OOM.

Historical throughput was 240 forwards in about 57 minutes. Linear extrapolation
puts 368 at roughly 87 minutes of inference; this is an estimate, not a benchmark
for these shorter prompts. Hard wall-time budget is 180 minutes across two
units, plus model-free build/preflight. Runtime budget does not authorize extra
forwards. No new model or paid compute is required by this design.

## Deliverables and current evidence

- `scripts/natural_binding_pilot.py`: corpus builder, exact job ledger,
  projection/control reference, discovery-only fit and row gate.
- `scripts/test_natural_binding_pilot.py`: synthetic numerical and integrity checks.
- `scripts/natural_binding_tokenizer_preflight.py`: reproducible tokenizer-only checks.
- `docs/research-data/natural-binding-causal-pilot-v1/`: corpus, forward ledger,
  tokenizer/metadata preflight, and model-free design checksums.

The original causal negative remains closed at `1b900054`; no old result was
relabelled, no holdout reopened, and no research claim was changed. This draft
makes a deliberately small, falsifiable bet. It could fail simply because the
single rank/site is inadequate; that limitation is accepted in advance.


## Model-free validation completed 2026-09-24

Seven synthetic tests pass, including additive-nuisance cancellation, recovery
of a planted interaction direction, preservation of the orthogonal residual,
matched perturbation norms, split isolation, exact ledger cardinality, and
rejection of wrong-sign or nonspecific effects. All 40 prompt encodings pass
the tokenizer-only preflight; all 16 distinct complete answers are single
tokens after the frozen newline boundary. Prompt lengths including BOS are
7–11 tokens. These are engineering checks, not model behavior evidence.

Reproduce from the branch root:

```bash
python3 -m unittest discover -s scripts -p test_natural_binding_pilot.py -v
python3 scripts/natural_binding_pilot.py --output-dir docs/research-data/natural-binding-causal-pilot-v1
uv run --with tokenizers==0.23.2 python scripts/natural_binding_tokenizer_preflight.py --vindex /mnt/models/larql-vindexes/gemma-4-e4b-upstream-20260811-q4k.vindex
```
