# Review of the P1 showcase draft — Codex, 2026-10-27

Reviewed **a1a20581e96a27b218a7e27935889be69d56fee5**,
`docs/phase-b-p1-writeup.md`, in `research/edit-certification`.
Line numbers below refer to that exact revision (142 lines). The draft
was not edited. No model or publication/push action was performed.

**Verdict: the numerical core checks out, but the draft needs scientific
scope and chronology corrections before publication.** The two concrete
C2 counterexamples remain useful after these corrections.

## Required corrections

### 1. Seven mandatory controls, not nine passes — line 7

The bench evaluates C0–C8, but certification requires **C0, C1, C2, C3,
C4, C6 and C7**. C5 and C8 are nonblocking. C5 is NOT_EVALUATED on every
P1 case, so “certifies ... only if it passes nine controls” describes a
different rule and misleadingly makes these cases look impossible solely
because no donor control exists.

Suggested wording: **“The bench evaluates nine controls, seven of which
must pass for certification; donor ceilings and propagation are reported
separately.”**

Evidence: the frozen design's control/verdict table and the independently
replayed C0–C8 statuses in 68f7fda9.

### 2. Disclose the pre-existing W1 gap and 3/5 ceiling — lines 17, 27, 44–54

BMW and Gretzky already lacked a qualified mandatory answer-only witness,
W1. Their native answers were “B.” and “after the legendary hockey
player,” respectively. **C3 could never PASS for these two on the fixed
question sets, even if C2 had passed.** This was identified before the
Codex solves, and the closure document already discloses it.

The actual HQ/country/nickname changes are valid, independent C2 failure
witnesses. They are not the sole obstacle separating these candidates
from full certification. The public draft currently drops this caveat
while presenting “refused because of collateral changes” as the complete
explanation.

Add the native coverage gap and the known **maximum of 3/5 possible full
certifications**. Say that C2 supplies an observed collateral failure,
while full certification was also blocked by missing native W1 coverage.
Do not imply that preserving those attributes would have certified BMW
or Gretzky. No rewording/requalification or new run is requested.

### 3. Restrict “every phrasing” to the qualified/included set — lines 14, 27, 75

Gretzky E1, “The sport of Wayne Gretzky is”, gives **“a game of numbers.”**
after the edit. It was excluded because its native answer did not state
the old value. Thus the claim that the new value appears in *every*
phrasing is false without the qualification scope.

Use **“every qualified efficacy phrasing”** for C1 and **“every included
S/E item in the probe”** for the probability statement. Report each
witness's denominators: ES 1/1 and PS 3/3, not four unseen successes.

Likewise, native qualification is a declared surface/negation/stop rule,
not a guarantee of semantic correctness. Shakespeare E2's “not known to
the British government ...” is the previously documented spurious
qualification and remains included. Replace “answers correctly” with
“qualifies under the frozen native-answer rule,” and mention that known
limitation rather than implying every retained native answer was correct.

### 4. Describe a ROME-inspired proxy, not the published evaluation — lines 9, 13, 17, 30, 74, 90–108

The literature numbers **100.0, 96.4 and 75.4** are correctly copied from
the ROME GPT-2 XL row of Table 4. The P1 measurement procedure and sample
are different. The
[official CounterFact evaluator](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/experiments/py/eval_utils_counterfact.py)
uses original evaluation prefixes and token-mean target NLL. Our probe
uses a teacher-forced native-answer prefix and summed suffix log scores.
Our NS also includes unrelated O4 controls and our own neighbour choices.

The [paper](https://arxiv.org/abs/2202.05262), §3.3/Appendix D, samples
neighbours sharing the original relation/object; it also explicitly
discusses essence drift. This does not equate its evaluation with C2,
but it makes a universal “CounterFact-style metrics never ask/cannot see”
claim too broad. Its published evaluation source also has an
attribute_prompts path; that alone is not evidence of an equivalent C2
preservation gate, and no such equivalence is claimed here.

Use **“our declared ES/PS/NS-inspired probability screen”** and mark the
paper column as historical context, **not a matched benchmark comparison**.
The supported conclusion is: **these two edits pass our included efficacy
and off-target tests while failing explicit same-subject attribute checks**.
Do not assert that the full published evaluation would score them as
successes: it was not run.

Similarly, “applied the way its authors applied it” and “reproduce a
published method to within 0.03%” should become **“two independent
implementations of the pinned ROME equations under our shared, declared
P1 adaptations agree in factors and update norms.”** The 20-update policy,
fp16, covariance and fixed template differences remain material context.

### 5. Limit Nintendo and mechanism claims to observations — lines 40, 80, 84

Only the **Codex adapters** were used for the archived certification and
probe model runs. Claude A3 independently produced nearby factors and a
similarly large norm. That does **not** establish that its greedy answers
were identical or also gibberish. Nor was the unmodified upstream ROME
implementation executed on this prompt.

Replace “Both implementations produce this, because it is what the
reference code does on this prompt” with: **“Both independent solves
produce an update norm near 1332; generations with the Codex adapter show
the reported failures.”**

The small denominator algebraically amplifies the output factor. The
subject occupying position zero is observed; its being the cause of the
small denominator was not established by an ablation. Replace “is the
first token ... so” with **“is at position zero; its measured clean-key
projection is 0.113, versus 6.905 for the mean key.”**

Finally, line 80 infers drift of the model's *whole entity picture* from
a handful of queried attributes. Say **“other probed attributes change”**.
The output associations are measured; the internal representation or
causal story behind “Italian BMW” and “Mexican baseball Gretzky” is not.

### 6. Correct the convention-declaration chronology — lines 118–122

“After the solves had run, but before any output was read” is not an
accurate common chronology:

- Claude's erratum **2373d9cd** says its conventions were declared while
  its A3 solves were running, before its A3 outputs/comparison were read.
- Codex's complete declaration **60482963** was committed after the A3
  solves existed and after Codex's own old solve results and first 0/5
  gate had already been read. It preceded **Codex's read of the new A3
  factor/status artifacts and the new comparison**.
- The later comparator **dd12dc99** was committed before its first
  execution. The initial update/source/threshold declarations already
  existed; the joint detailed numerical convention disclosure was later.

Give that participant-specific chronology. “Before any output” wrongly
suggests that these were prospectively declared before all solve-result
inspection. The explicit post-run disclosure is part of the evidence,
not a reason to hide or erase those runs.

### 7. Replace “random changes nothing” with the measured criterion — lines 16, 76

The random counterpart does not flip the tested comparisons on these two
witnesses. Its probabilities are not unchanged: BMW E3's margin changes
from **−6.09374994 to −6.00000018**, a shift of **0.09374976**. The included
random greedy answers match native answers, but attributes and off-target
free generations were not all run under the random condition.

Use **“no target flips on the included probability probes, and no change
to the tested random-condition greedy answers”**, rather than a universal
zero-effect claim.

### 8. Bound the claim about my independent verification — lines 115–117

The allocation of work—covariance, the other solve, independent control
replay and probability arithmetic—is accurate. The cross-checks did not
independently re-run all model probabilities or recompute the published
paper results from their raw experiments. C4 uses archived margins;
the probe arithmetic uses archived token log probabilities.

Replace “Every published number was recomputed ...” with a precise scope:
**“Both agents independently recomputed the factor-comparison metrics,
and Codex replayed all 45 control statuses and 111 probe-margin sums from
the archived outputs, without importing the bank's judges.”** Literature
values were checked against their primary source, not independently
reproduced. The expected P1-5 headlines were known before my audit, so it
was independent implementation, not a blind assessment; the closure
already makes this distinction.

## Smaller factual corrections

- **Line 96:** the pre-registered proxy also requires **NS >= 75%**.
  Include all three conditions. This does not change the failed reading,
  because NS=13/16 passes while ES and PS miss their thresholds.
- **Line 125:** the double KL coefficient was in a **loss / early-stopping
  calculation**, not a budget argument. The budget error was a separate
  under-estimate of P1-5's plan. Keep those incidents distinct.
- **Line 20:** “mixed” is Pat's closure description for an outcome not
  covered by the original reading table; it was not itself a pre-registered
  table row. “A mixed outcome not covered by our pre-registered table” is
  precise and consistent with the closure decision.
- **Line 58:** label the table as **answer excerpts**. For example,
  “Germany in 1885.” is shortened to “Germany”, and the native nickname
  answer continues after “The Great One.” The displayed factual values,
  including BMW's propagation question at line 66, are present in records.

## Numbers and observations that check out

- 0/5 certified; all five verdict/control summaries; all 45 control statuses.
- A3 5/5 agreement: minimum U cosine **0.9999907809**, largest norm
  difference **0.02743748%**, correctly rounded to 0.03%.
- Initial disagreement: U cosines 0.59–0.75 and norm ratio up to 424;
  objective factors 1.7–6.9 and Nintendo key factor about 61.
  The factor identity holds to stored-array rounding (maximum relative
  discrepancy 3.28e-8), not necessarily exact serialized equality.
- The listed marginal ranges (native about −8.1 to −3.1, edit +5.8 to
  +10.7) and the Renault/Michael Jordan examples are correctly rounded.
- All quoted question/answer values, Ford's excluded outcome, and the
  Germany/Nintendo failure examples are supported by the raw records,
  subject to the scope qualifications above.
- Probe 3/4, 12/17, 13/16, and random 0/4, 1/17, 15/16.
- **9938** forwards since v2 and **13364** approved; the headline is
  explicitly scoped to v2, so omitting the separate 500 v1 forwards here
  is not a numerical error.
- Primary-source Table 4 values **100.0 / 96.4 / 75.4** are correct;
  their direct comparability is the issue, not transcription.

The certification/probe archive and both live manifests were rechecked:
**114 payload hashes verify and archived bytes match live bytes**. No
primary numerical discrepancy was found in the checked headline figures.

## Suggested core claim to retain

“On two fixed GPT-2 XL cases, independently constructed rank-one adapters
pass our qualified efficacy and off-target tests while changing other
queried attributes of the same subject. C2 exposes those collateral
changes. These cases also have native coverage gaps, so they were not
eligible for full certification on this question set. Across all five
cases, the bench certifies none and the global probability-proxy criterion
is not met. The cause of the earlier Gemma negatives remains unresolved.”

This retains the showcase's concrete result without claiming a full
CounterFact reproduction, a new causal explanation, or prior complete
certifiability of the two witnesses. No new experiment or looser gate is
needed to make these wording corrections.

The [review receipt](research-data/phase-b-p1-writeup-review-codex.json)
pins the exact draft, numeric checks, counterexamples and evidence hashes.
This is a documentation-only review: no solver, tests or measured outputs
were changed; no model tests were necessary or run.
