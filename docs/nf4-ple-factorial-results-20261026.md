# NF4 residual × PLE causal test — 26 October 2026

**The frozen number endpoints do not support H2 on this case.** Replacing the
entity's projected PLE inputs moves two−one by only **+0.1445** without the
residual intervention and **−0.1836** conditional on it. Both fail the practical
threshold of +1 logit. The intervention is active: PLE alone moves He−H by
**+9.6719**, while its matched random control moves it by +0.0586.

This is a narrow negative for the tested PLE transplant, not a general claim
that PLE is irrelevant or that name-form H1 is correct.

## New mandate and frozen design

The previous [asymmetry series](nf4-asymmetry-results-20261026.md) observes a
large margin asymmetry but fails its operational top-1 gate. That negative
gate remains unchanged. A new operator mandate authorizes this margin-based
test. Protocol, scheduler and checker are committed at **`ae02bfb1`** before
any PLE intervention on the pretrained model.

The native value boundaries, helium donors, token readouts and residual vector
are exactly those already saved. Symbol uses bare H/He inside LaTeX; number
uses the native words ` one`/` two`. No new question, gain, donor or layer is
selected. All primary input ids are held fixed except in the explicit full
donor control. These are resumed native-prefix diagnostics, not new integrated
edited responses from empty.

Two factors are independently set:

- **R:** the stored hydrogen→helium entity-row vector, once post-L21 at row 5,
  with its unchanged norm rule and amplitude.
- **P:** replace the final projected PLE input at entity row 5 in **all 42
  blocks** by the saved complete-helium donor input. The input ids and main
  embedding remain native. Other rows are untouched.

The final PLE input combines a token lookup and a projection from the main
embedding. This intervention replaces that combined input; it does not isolate
the lookup component alone. R+P also does not reconstruct the entire donor
state: early residual/attention history and main embeddings still differ.

For each attribute, the eight conditions are native, R, self-P sham, P, R+P,
random P, R+random P and full donor. Two supplied-prefix off-target symbol
questions, lithium/carbon, each add native, R+P and R+random P: **22 forwards**.

Random P is a single Gaussian draw, seed 20261027, scaled separately at each
layer to match the donor-minus-native PLE change norm. The actual rounded
float16 additions must match those norms within 0.1%; the observed maximum
relative mismatch is **0.00959%**. No normalization search or seed selection
follows the results. The old random-R evidence is retained as source context,
not counted as a new control run.

## Endpoints and results

Let m00, m10, m01 and m11 be target-minus-native logit margins for native, R,
P and R+P. The literal H2 endpoint is m01−m00 on the number question. The
conditional endpoint, relevant to the earlier R setting, is m11−m10.
Each passes only if it is **at least +1 logit** and **greater than twice the
positive part of its matched random-P effect**. This is a predeclared practical
criterion on one case, not a statistical significance test.

| Condition | Symbol He−H | Number two−one |
|---|---:|---:|
| Native | −21.2656 | −18.0430 |
| R | −1.5313 | −17.9961 |
| Self-P sham | −21.2656 | −18.0430 |
| P | −11.5938 | −17.8984 |
| R+P | −2.9063 | −18.1797 |
| Random P | −21.2070 | −18.4590 |
| R+random P | −1.3594 | −18.6367 |
| Full helium donor | +1.9688 | +2.7188 |

| Effect | Symbol | Number |
|---|---:|---:|
| P without R | **+9.6719** | **+0.1445** |
| Matched random P without R | +0.0586 | −0.4160 |
| P conditional on R | **−1.3750** | **−0.1836** |
| Matched random P conditional on R | +0.1719 | −0.6406 |
| Factorial interaction | −11.0469 | −0.3281 |

The PLE transplant is directionally active on the symbol. Its effect is strongly
non-additive with R on this example: adding P to R reduces the He−H margin.
For number, the small positive effect without R is real in the saved logits
but below the frozen effect-size threshold; with R, the effect points away
from two. Neither number endpoint passes.

Native, R and full-donor full-logit replays are bit-identical to the previous
captures; both self-P shams are bit-identical to native. The complete donors
still select He and two, so target accessibility is not the limiting factor.
Every primary P/R+P condition keeps native H or one as top-1. The predetermined
release phase is therefore skipped: **zero continuation forwards**. No edited
free answer or persistent fact edit is claimed.

Lithium and carbon keep Li/C as top-1 in all three off-target conditions.
Their value prefixes are explicitly supplied. This covers two local examples,
not an off-target failure-rate estimate or full-chat selectivity.

## Execution, lock coordination and verification

- One model service performs 22 forwards in 45.0 s, exiting zero, under the
  unchanged 22G high / 24G maximum / 8G swap / 30-minute limits. Its final
  service log reports 9.4G memory peak and no swap.
- Planned maximum is 38 forwards across core and conditional release services,
  with individual caps of 24 and 16. The release service is not launched.
- The synthetic prelaunch fixture runs the actual 22-condition scheduler,
  all four changed-case releases, and the 16-forward truncation path, alongside
  exact hook-support/norm/zero/cap checks. It remains labeled synthetic and
  cannot be used as a real source by the production path.
- After the core service is launched, the operator requires a shared heavy-model
  `flock` for every subsequent NF4/LARQL service, including other worktrees.
  The existing service is finished when checked. No later model service is
  needed here. The rule is recorded in [AGENTS.md](../AGENTS.md), commit
  `3f194731`; this report does not retroactively claim the core held that lock.
- The model-free checker returns **`verified_complete`**, independently of the
  negative scientific endpoints. It recomputes margins, factorial effects,
  endpoint readings, rowwise random norms, exact float16 additions, source and
  sham replays, hashes, argmaxes and cgroup bounds.
- Runner and checker are written by the **same coding agent**. This is not
  independent verification or replication.

Sources: [`run_nf4_ple_factorial.py`](../scripts/run_nf4_ple_factorial.py),
[`check_nf4_ple_factorial.py`](../scripts/check_nf4_ple_factorial.py) and
[compact evidence](research-data/nf4-ple-factorial-20261026.json). Raw evidence
is local under `nf4-ple-factorial-20261026/factorial`. The established outputs
directory holds a ZIP with a SHA-256 manifest, source evidence, frozen code
and tokenizer; restoration is checked without loading model weights.
Metadata and code are hashed; full weight shards are not fully hashed.

## Claim boundary and next decision

H2's proposed sizeable number movement is not supported by this full projected
PLE replacement, either alone or with the fixed R. This does not prove zero
PLE influence: the +0.1445 number change is measured, just below the stated
criterion. Nor does the result establish H1 or a general architectural limit.

The full donor changes main embeddings and the causal residual/attention
history as well as PLE. Those remaining differences are not isolated here.
Relate this negative to the separately running H1 study before choosing the
next bounded test; no new layer, gain or donor search is added to this mandate.
No model weights, vindex or persistent patch change, push or publication occur.
