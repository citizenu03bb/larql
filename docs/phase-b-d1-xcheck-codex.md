# Independent D1 recomputation — 2026-10-27

**The numerical headlines reproduce within the declared tolerance.** The
forced probe is invalid (Bolt L12: 1/4 positive unseen margins, required
3/4); no validated efficacy classes follow. The dose/bleed AUC is
**0.6923076923**, inconclusive. Neither I nor W meets the dose-index
improvement threshold. There is one substantive verification limitation:
**M3's served outputs were not archived**, so its error magnitudes cannot
be independently reconstructed from vectors. Their recorded thresholds
and the separate rank-one algebra were checked.

This cross-check ran **zero model forwards**, made no download, and did
not begin P1, launch D2, reopen B2 or access the sealed test set. No push
was made. Existing method files and B2 verifiers/results are unchanged.

## Inputs and independent method

The specification is the frozen D1 protocol, with plan
`50668f2038cf39d5453af2c29174a045c6b31e3703a16cd8a5f62fb219f9d9e1`.
The six service receipts, five `keys.npz` files, W covariances and original
15 edits are inputs. The code imports neither Claude's runner/analysis
nor Codex's solver/covariance implementations. Their relevant hashes are
checked instead. Expected headlines were already supplied in Pat's brief;
this is not claimed as a blind freeze.

`tools/rome_solver/xcheck_d1.py` independently solves the six C/W linear
systems in float64, with five right-hand sides per layer, and forms the
identity direction directly. It checks SPD matrices and normalization.
For C, the stored actual V is checked against the independent solve and
then used. I/W directions are cast to the existing float32 adapter format;
projections accumulate in float64. Margins are recomputed by summing the
archived per-token log probabilities from the recorded divergence point.
AUC uses positive/negative pair comparisons with half credit for ties;
Spearman uses average ranks for ties. Three model-free tests cover these
statistics, minimum label counts, class boundaries and order requirements.
All pass.

Across **1,854 numerical comparisons**, 1,152 are exactly equal and the
largest absolute difference is **2.9802226e-7**. None exceeds the declared
absolute 2e-6 / relative 1e-5 tolerances. Small floating-point differences
are retained; values were not adjusted to fit the peer output. There is
no numerical, categorical or label discrepancy in the replayed fields.

## Recomputed observations

| Quantity | Independent result |
|---|---|
| Archived forwards | 1,342 across six successful services, exactly their planned counts |
| C1-positive probe checks | Beethoven L8 4/4, Beethoven L12 4/4, Toyota L8 4/4, Bolt L12 1/4 |
| Global M2 validity | false |
| Median unseen subject dose, arm C | 0.7319969 to 1.1962391 across the 15 edits |
| Smallest individual unseen subject dose | 0.6797024 across all 60 E entries |
| Same-type neighbour dose, C | median 0.7100511; range 0.0330767 to 0.9122562 |
| Unrelated O4 dose, C | median 0.2282904 |
| Dose/bleed pairs | 13 bleed, 23 kept; labels checked against v2 generated text |
| AUC | 0.6923076923; inconclusive under 0.6/0.8 thresholds |
| Median absolute random E shift | 0.0625001 nat; below 1 nat |

All unseen subject doses exceed the registered 0.5 key-dose threshold.
This rules out **K as defined by that threshold on these captures**. It
does not establish a value-side explanation, or prove that key geometry
is irrelevant to every kind of efficacy failure. V/S/P/U remain unvalidated
because M2 fails its global validity gate.

The raw numerical dose/shift Spearman value also reproduces
(0.4200772), but its margin-based interpretation is suspended with M2.
Claude's JSON retains intermediate `class`/`classes_C` values despite the
invalid probe. Their mechanical calculation reproduces; they must not be
used as validated efficacy classes. The independent output explicitly
separates `mechanical_class` from an empty `validated_classes_C`.

## Covariance comparison and D2

For both I and W, **zero of the four C1-positive edits** has a selectivity
index at most 0.67 times C. Zero reaches the 1.5-times worse threshold.
Thus neither arm can satisfy the registered improvement conjunction:
its dose-index part already fails, independently of the invalid flip
probe. The mechanical category `neutral` therefore has M1-only support
in this dataset; no M2 conclusion is needed to rule out the trigger.

This is neutrality under the **registered key-dose index**, not proof of
equal behavioural selectivity or certification performance. No D2 trigger
is reached. The AUC is inconclusive, so the separate protocol branch
requiring the dose to predict bleed is not reached either. High neighbour
doses are observed with and without bleed. Statements that bleed depends
*mostly* on edit strength rather than keys remain hypotheses, not an
identified causal decomposition from D1.

## M3: what is independently checked, and what is missing

The records contain `m3/L*/native_rel` and per-arm `rel` scalars only.
Neither the unedited nor edited served output vectors are present in
the five key archives or the remaining manifested files.

- The 15 recorded native errors are below the 2% threshold; their maximum
  is **0.0005167641** (0.0516764%).
- All 45 recorded edited errors are below the 5% threshold; their maximum
  is **0.0012942711** (0.1294271%).
- Independently, the saved U/V/k*/Wk*/v* algebra gives a largest normalized
  discrepancy of **3.5142064e-8** across the 45 arm/edit combinations.

The last item checks mathematical consistency, not the actual served NF4
module. The first two recompute a **threshold decision from recorded
errors**, not the errors themselves. Consequently the correct closeout
is: *reported M3 passes, thresholds independently checked, numerical
served-module error not independently reproducible from this archive*.
This limitation does not prove an implementation fault; it prevents a
claim of complete cross-verification of M3. No missing output was guessed
and no model/module call was made to fill the gap.

## Integrity and reproduction

All 33 payload hashes and the additional manifest match the zip and live
files. Archive SHA-256:
`14877f775d2ab6134239b83f154aa3a29827ca0b622a9887526ce5029bb3b606`.
This matches the prefix/suffix announced in the brief; there is no
separate zip checksum sidecar in the output directory. The CRC check
passes. The delivered root has no `index.json`; verification instead uses
the complete plan, six status/record pairs and `MANIFEST.sha256`.

The 150 captured arrays have the expected prompt lengths and last-subtoken
subject positions under the pinned local tokenizer. All 42 continuation
token sequences, first divergence positions and native-answer prefixes
were reconstructed independently, with no difference. The three excluded
neighbour probes indeed have true values overlapping the new value.

The W service pins the unchanged covariance source, 200 windows, 12,800
positions per layer and the three resulting matrix hashes. D1 does not
archive the WikiText per-position activations, so this cross-check verifies
the matrices' provenance/shape/SPD properties and their downstream use;
it does not recreate their model-based estimate. Likewise, recorded token
log probabilities are summed and audited, not regenerated from logits.

Receipts: [summary](research-data/phase-b-d1-summary-codex.json),
[full arithmetic replay](research-data/phase-b-d1-xcheck-codex.json),
[tokens and prefixes](research-data/phase-b-d1-token-prefix-audit-codex.json),
[archive](research-data/phase-b-d1-archive-audit-codex.json),
[M3 coverage and algebra](research-data/phase-b-d1-algebra-coverage-audit-codex.json).
The first replay and its SHA-256 remain pinned in the versioned output.

Reproduction, using a fresh output path and no model:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /mnt/models/finetune/venv/bin/python tools/rome_solver/xcheck_d1.py \
  --root ~/Projects/larql-research/phase-b-d1-20261027 \
  --out /tmp/d1-independent-NEW.json
```

Pat's C7 arbitration is acknowledged as prospective from P1-5; historical
B2 outputs remain intact. Codex's P1 port and covariance work have not
started and still require the separate P1-2 decision. The independent
Claude P1 solve directory was not read.
