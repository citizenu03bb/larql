# Independent edit-certification replay — frozen result

This result was written from Pat's R1–R5 brief and the raw archived runs,
before reading the certification harness, its design document or its replay
reports. The script imports only NumPy and the Python standard library. It
does not import any archived runner or checker. No model was loaded or run.

The machine-readable result is
[`research-data/edit-cert-replay-xcheck-codex.json`](research-data/edit-cert-replay-xcheck-codex.json).
The verifier is [`../scripts/xcheck_edit_cert_replay.py`](../scripts/xcheck_edit_cert_replay.py).
All five edits are **REFUSED**. This is an independent verification of the
replay arithmetic and certification rules, not an independent replication of
the experiments that produced the archives.

| Case | C0 | C1 | C2 | C3 | C4–C8 | Verdict |
|---|---|---|---|---|---|---|
| R1, single residual pulse | FAIL | FAIL | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | REFUSED |
| R2, static g=1.0 | PASS | NOT_EVALUATED | FAIL | FAIL | NOT_EVALUATED | REFUSED |
| R3, random pulse as edit | FAIL | FAIL | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | REFUSED |
| R4, full subject donor | FAIL | NOT_EVALUATED | FAIL | FAIL | NOT_EVALUATED | REFUSED |
| R5, Q4K residual pulse | FAIL | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED | REFUSED |

Every specified native frontier qualifies: NF4 symbol H, number one, and
Q4K number/mass 1 are native top-1. The two off-target native top-1s are
lithium 17623 and carbon 236780. No resumed frontier can establish PASS
outside C0. All efficacy formulations here were already seen, so C4 has
no eligible unseen efficacy question. There is no reloaded generation for
C7 and no propagation test or failure rule for C8 in this composition.

The raw recalculation reproduces all 48 numeric references exactly after
the two explicit rank conversions below. Of these, 46 also equal the raw
archived scalar without any conversion. All 24 supplied NPZ SHA-256
references match. The six Q4K result NPZs have no supplied result digest;
their current hashes are recorded, without claiming historical validation.
All numbers and reference locations are retained in the JSON.

| Observation | Recomputed value |
|---|---|
| R1 symbol margin, native → residual | −21.265625 → −1.53125; increment +19.734375; top-1 H |
| R1 number margin, native → residual | −18.04296875 → −17.99609375; increment +0.046875; top-1 one |
| R1/R3 random symbol / number increments | +1.056640625 / −0.0234375 |
| R1/R3 off-target edited top-1s | lithium 17623; carbon 236780, both unchanged |
| R2 complete g=1 symbol / number margins | +25.93212890625 / −0.5; top-1 He / three |
| R2 row5 symbol / number increments | +19.734375 / +0.046875; top-1 H / one |
| R2 row30 symbol / number increments | +45.0126953125 / +15.58203125; top-1 He / three |
| R3 float16 residual / random vector norms | 63.89672501632937 / 63.89700397455939 |
| R4 donor symbol / number top-1 | He / two |
| R5 number margin, native → edit | −20.340023040771484 → −19.581012725830078 |
| R5 mass margin, native → edit | −26.873004913330078 → −21.302377700805664 |
| R5 number random increments, seeds 20261025 / 20261026 | −0.11541557312011719 / −0.061492919921875 |
| R5 strict-greater one-based mass-4 rank, native → edit | 48 → 4 |

Interpretations fixed for this verifier:

- C0 concerns the candidate's application rule, not the procedure used to
  measure its archived frontier. R2's static lexical hook runs at every
  forward; matching a generated Hydrogen token is not an answer-time
  schedule. It requires neither a native proposal nor a fresh donor pass.
  The donor passes used to construct/calibrate a vector are distinguished
  from donor inference required to apply that fixed vector.
- A unique stored artifact means the numerical vector with fixed protocol
  parameters. It does not establish a deployable, self-describing patch.
  The pulse cases have such a vector but fail the native-proposal,
  response-time scheduling and every-forward requirements. R4 is an
  observed donor substitution at a saved answer boundary, not a stored
  residual rule; no residual layer is written. Its donor is the edit,
  not a C5 ceiling.
- Only g=1.0 is the complete R2 edit. The specified row5 and row30 splits
  are partial witnesses; other gains and both_half are outside this case.
  Partial symbol failures do not count against C1. Number top-1 three
  counts against C2/C3, regardless of whether two overtakes one. Complete
  number failure is grouped before by question order; row30-only failure
  is grouped after. Consequently R2 has C3 failures in both groups.
- R5 uses the session's precise gain 2.585096597671509, rather than the
  rounded 2.585 in the brief. Its archived mass ranks 47 and 3 are
  zero-based counts of strictly higher logits. Under the stipulated rank
  definition they become 48 and 4. Raw exact equality is false for those
  two scalars; equality after the documented +1 conversion is true. This
  is a reported convention difference, not an alteration of the archive.
- Both margin subtraction and increment subtraction use float32.
  Every 3b increment uses that archive's own native logits. Float16 norm
  means a float64 L2 accumulation after casting the vector to float16;
  the relative mismatch is approximately 0.000004366, below 0.001.
  The random norm also exactly equals the norm recomputed from the
  archived random NPZ's requested vector; no archived norm scalar is
  invented for it. That requested vector equals the stored float32 random
  vector after a float16 round-trip, not before casting; both equality
  checks are recorded explicitly.
- C5/C8 are nonblocking. The brief supplies no applicable negative ceiling
  or propagation criterion for these cases; their status is
  NOT_EVALUATED. Donor numbers may be reported without establishing PASS.

Validation before freeze: precision/tie/qualification/partial-mention
self-checks passed; a complete replay read 30 NPZs and reproduced the
numeric references above. The unrelated uncommitted L14 partition runner
was not edited. Reproduce without inference, using an environment with
NumPy:

```sh
python scripts/xcheck_edit_cert_replay.py --self-test \
  --archive-root "$HOME/Projects/larql-research/exploratory-natural-fact-edit" \
  --out /tmp/edit-cert-replay-xcheck-codex.json
```

This report and its JSON are the pre-harness freeze. Any subsequent
black-box comparison belongs in a separate report; discrepancies must be
left for Pat to adjudicate rather than used to revise this result.
