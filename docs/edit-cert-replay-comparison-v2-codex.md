# Comparison v2: A1/A2 and baseline regression

Pat authorized a separate v2 after the original generator's recorded failure
in `05768fb6`. The v2 reads only the pinned frozen JSON from `5b85fba8` and
the six emitted harness JSON reports, plus its own source for provenance.
No harness execution, model pass, raw experiment replay or push occurred.

[`../scripts/compare-codex-v2.py`](../scripts/compare-codex-v2.py) is versioned
here and copied byte for byte as `compare-codex-v2.py` in the A1/A2 archive
folder. The original `compare-codex.py`, its failure, the freeze, the first
comparison and both files from `05768fb6` remain unchanged.

The mapping table contains the original 40 exact quantity/location pairs,
unchanged, and exactly five explicit additions:

- R1 symbol donor top-1 → `donor_ceiling_observations.symbol.top1`.
- R3 lithium native top-1 → `native_top1` of the lithium observation.
- R3 lithium random top-1 → `numbers.lithium_random_argmax`.
- R3 carbon native top-1 → `native_top1` of the carbon observation.
- R3 carbon random top-1 → `numbers.carbon_random_argmax`.

There is no inferred quantity-name mapping or fallback. An unknown,
duplicate or missing harness quantity aborts without producing an output.
Existing output files cannot be overwritten. These four failure conditions
were exercised using temporary copies, without changing archived reports.

| Existing reports | Values | Verdicts | Control statuses | Qualification differences | Form differences |
|---|---:|---:|---:|---:|---:|
| `f6855066`, baseline | 40/40 | 5/5 | 45/45 | 2, R3 LI/C | 3, R4 C0 |
| `620a50b0`, A1/A2 | 45/45 | 5/5 | 45/45 | 0 | 3, R4 C0 |

Every verdict is REFUSED. For the baseline, **all case comparison payloads,
the entire summary and the six report hashes equal `64961ae8` exactly**.
This includes the original qualification and form differences, not merely
their counts. Generator metadata identifies v2 and therefore differs.

The A1/A2 run completes successfully with all 45 values exactly equal to
the freeze and the reported archived references. R3 LI/C both qualify as
`boundary`. R1 C5 reports the seen symbol donor top-1 He (2209) and stays
NOT_EVALUATED. The three R4 C0 form differences remain open for Pat:
response-time trigger true/false, native-proposal use true/false, and
every-forward application false/true (Codex/harness). C0 stays FAIL under
both readings. No new discrepancy was found or corrected.

Artifacts:

- [A1/A2 v2 output](research-data/edit-cert-replay-blackbox-comparison-A1-A2-v2-codex.json).
- [Baseline v2 output](research-data/edit-cert-replay-blackbox-comparison-v2-regression-codex.json).
- [Regression, rejection tests and preservation receipt](research-data/edit-cert-replay-comparison-v2-validation-codex.json).
- [Original and v2 SHA-256](research-data/edit-cert-replay-comparison-v2.sha256).

The original hash is
`2593968f3866fbf5001446f5879ad22310c8cbca24f1eb891c87d5214e3adeb6`;
v2 is
`b2ee67d2935b61eb1e9481a9f3c0113b65e68dd57faa957ec735e43476079268`.
The same outputs and `comparison-v2.sha256` are archived alongside v2 in
`edit-cert-replay-blackbox-codex-A1-A2-620a50b0-20261026/`.
The independent validation runner, `validate-v2.py`, is also archived there
and hashed in the validation receipt.

Reproduction uses `python3 compare-codex-v2.py --freeze <frozen-json>
--reports <existing-report-folder> --harness-commit <full-commit> --out
<new-output.json>`, with full commit `620a50b081ec6ad723121fd2591507d610def5e5`
for A1/A2 or `f6855066ff9f7117da26a06bf0b7454ccfcdccec` for the baseline.
These checks establish agreement on these archived replays; they make no
claim of persistence or an edit of a fact.
