# P1-5 independent replay and reading — Codex, 2026-10-27

**Recommendation: keep the causal attribution of B2 suspended, and close
P1 as a mixed result if Pat agrees.** The B2/P1 measured refusals remain
valid observations. “Suspended” concerns interpreting B2 as a failure
specific to Gemma E2B/E4B or quantization; it does not erase those results.

The frozen pooled P1 reading is not met. A per-fact account is useful as a
clearly posthoc descriptive analysis; it cannot replace that reading.
Moreover, only **BMW and Gretzky**, not all three C1-positive cases, meet
the probability proxy when its thresholds are applied per fact.

## Independent verification

This check uses the raw records and archived token log probabilities,
Codex's independently written control implementation, and the probe rules
at **3205f778**, read as text. No harness decide/judge or peer verifier is
imported, and no pretrained weights or model forward are used. Expected
headlines were supplied in Pat's brief 46, so this is not a blind audit.

- Zip SHA-256 matches **d82c89c716e7ba395cfe585d381b65642c540e26b40bb26e9350947e11a2ebc0**.
- Both manifests verify: **108 certification files and 6 probe files**.
  Every listed live payload is byte-identical to its archived copy.
- All five specs, committed plans (**dee168bb**), edit content hashes,
  saved/reloaded adapters, record/status hashes and service identities
  match. The probe uses its committed item plan (**41779bca**).
- All **45 control statuses, five verdicts and 63 native qualifications**
  agree with the bank. There are **zero discrepancies**.
- All **111 probe margins** equal the recalculated sums exactly. All six
  pooled metric rates/counts, validity and paper-proxy reading equal
  Claude's saved analysis exactly.
- Generations decode from the recorded token IDs; stop/cap flags match
  the local tokenizer's 180 stop tokens. Actual forward counts equal
  generated token counts plus diagnostic boundary calls.

The result is **0 CERTIFIED / 5**, **686 bench forwards in 18 services**,
plus **222 probe forwards**. Every service stays below its committed
ceiling. Spending since P1 v2 is **9938**, below the approved **13364**.
The certification index's per-fact forwards fields hold *plan ceilings*,
not actual consumption; this audit uses records for actual usage.

The BMW driver incident is preserved: service 0 is present once, completed
normally, at **39/312 maximum** forwards. There is no duplicate/retried
service in this inventory. The wrongly enforced equality was a driver
error; it does not turn this normal completion into a model-service
failure. The continuation retained this service and started later ones.

## The fixed pooled reading

The probe's committed rule explicitly says **pooled over the five facts**:
ES = 100%, PS >= 75%, and NS >= 75%, with every included native S/E margin
negative. The native validity check passes.

| Metric | Edit | Random counterpart |
|---|---:|---:|
| ES | 3/4 = 75% | 0/4 = 0% |
| PS | 12/17 = 70.59% | 1/17 = 5.88% |
| NS | 13/16 = 81.25% | 15/16 = 93.75% |

ES and PS miss their thresholds. NS meets its threshold. Therefore the
frozen “like the paper” **proxy is false**, and row 3 cannot be selected
for the overall P1 result. The source's NS includes O4, an unrelated
control, alongside same-relation neighbours; that definition is retained.
These small fixed-set proxy scores are not a reproduction of the paper's
full evaluation dataset.

Row 4's condition of **no probability-level efficacy** also does not
literally describe these observations. The two independent A3 solves
agree, and several facts show clear efficacy. It would be another posthoc
change to interpret “none” as “not enough for the pooled row-3 threshold.”
Rows 1/2 require certified edits, of which there are none. I therefore
accept Claude's **“no global row covers this mixed outcome”** reading.

## What can be said per fact

The following is a **posthoc decomposition**, not a new certification or
rescoring of the global gate:

| Fact | C1 | C2 | C6 | ES | PS | NS |
|---|---|---|---|---:|---:|---:|
| Bmw | PASS | FAIL | PASS | 1/1 | 3/3 | 3/3 |
| Germany | FAIL | PASS | FAIL | 1/1 | 0/4 | 4/4 |
| Gretzky | PASS | FAIL | PASS | 1/1 | 3/3 | 4/4 |
| Nintendo | FAIL | FAIL | FAIL | 0/1 | 4/4 | 1/4 |
| Shakespeare | PASS | NOT_EVALUATED | PASS | not measured | 2/3 | 1/1 |

- **BMW and Gretzky provide two concrete witnesses of the bench's added
  strictness.** They pass free-generation efficacy and the declared
  off-target controls, and meet all three probability-proxy thresholds
  when evaluated separately. They nevertheless fail the mandatory
  same-entity attribute controls: Munich becomes Milan; Gretzky's country
  becomes Mexico and his nickname changes. This narrow observation is
  directly supported without assigning a new global row.
- **Shakespeare cannot be called a third full “row 3 by fact.”** Its S1 is
  excluded from the probe, so ES is unmeasured. PS is **2/3**, below 75%.
  Its only NS item is **O4**, an unrelated control; none of O1–O3 is
  included. Its C2 is unknown because the edited A1 hits the cap, not a
  demonstrated preservation or the same evidence as the explicit BMW/
  Gretzky attribute failures. C1 still passes under the declared free
  generation rule. These measurements are compatible because the probe
  scores teacher-forced native-answer prefixes rather than the same free
  continuations. Shakespeare E2 is the previously declared spurious
  qualification; it is retained, not excluded after seeing its margin.
- **Germany and Nintendo miss the proxy differently.** Germany has seen
  efficacy but 0/4 positive unseen margins. Nintendo has 4/4 positive
  unseen margins under native-prefix teacher forcing, yet fails seen ES,
  neighbour preservation and free generation. These are further reasons
  to keep the different measurements distinct.

Thus I agree with Claude's global caution, but qualify the phrase “the
probe looks like the paper on BMW, Gretzky and Shakespeare”: **only the
first two satisfy every applicable numerical threshold in the frozen
proxy; the third has missing coverage and a PS below threshold.** No
value, question, denominator or threshold has been repaired.

## Scope and next decision

The independently replicated construction gate reduces the concern that
one implementation alone caused these GPT-2 candidates. It does not
validate the earlier Gemma optimizer, identify architecture/quantization
as the cause of B2, or establish that the bench can certify a real edit.
Nor does a failed certification prove an edit changes nothing.

For the showcase, the useful positive result is already concrete: the
bench detects **target efficacy with collateral attribute changes** in
two replicated GPT-2 constructions. Publish those examples with the full
0/5 outcome and the two outright failures. Keep Shakespeare's unknown
attribute result explicit. That communicates more than either “everything
failed” or a retrospective global success label.

I recommend no additional tuning or model passes to force a reading.
Pat can close P1 as “mixed outcome; global positive-control criterion not
met; B2 causal attribution unresolved.” This audit itself does not make
that closure decision or authorize another experiment.

## Reproduction and limits

Run `tools/rome_solver/xcheck_p1_5.py --out <new-output.json>` in the local
Python environment. It uses the committed probe plan exactly, without
importing Claude's planner or reselecting items after outcomes.

The [full audit](research-data/phase-b-p1-5-xcheck-codex.json) retains all
controls, per-question witnesses, 111 exact margin comparisons and per-fact
rates. The [comparison receipt](research-data/phase-b-p1-5-peer-comparison-codex.json)
checks Claude's six reported pooled rates/counts and validity.

The archived per-token log probabilities support arithmetic reproduction,
not an independent inference run. C4 likewise uses recorded boundary
margins; full logits are not available to rederive them. Random adapter
execution is evidenced by the archived records, not replayed with a model.
No claim of persistence or selective fact editing follows. No new model
service, blind-set access, candidate modification or push occurred.

All **16 targeted P1 tests pass**, including the new strict-zero-margin,
missing-ES and inclusive-75-percent checks. The A3 comparison and handoff
files remain byte-identical to 7a3824a5.
