# Edit-certification black-box comparison after the independent freeze

The independent verifier, JSON and report were committed together as
`5b85fba8f6a147aeae88b30ba813c8dba3081b9c` before reading any harness
report. The harness was then invoked, without reading its code or design,
from its worktree at `f6855066ff9f7117da26a06bf0b7454ccfcdccec`:

```sh
cd "$HOME/Projects/larql.worktree/edit-certification/tools/edit_cert"
python3 -m edit_cert replay \
  --out "$HOME/Projects/larql-research/exploratory-natural-fact-edit/edit-cert-replay-blackbox-codex-20261026"
```

Exit code: 0. Its `exit_criterion_met` is true. It reports reproduced
counts R1 14/14, R2 13/13, R3 3/3, R4 2/2, R5 8/8. This self-reported
exit criterion was not used to decide agreement: the comparison below
matches actual verdicts, controls and numbers against the frozen output.

| Comparison | Exact agreement | Differences |
|---|---:|---:|
| Verdict, in both harness reports | 5/5 | 0 |
| Control status, C0–C8 | 45/45 | 0 |
| Harness numeric/boolean replay items | 40/40 | 0 |
| C3 before/after group status | 10/10 | 0 |
| Qualification level | 16/18 | 2 |
| C0 form properties | 32/35 | 3 |

All five verdicts are REFUSED. Every harness numeric item also exactly
equals its harness archived reference. This includes mass-4 ranks 48 and
4; the harness makes the same one-based conversion as the independent
verifier. Twelve additional independent numeric items have no harness
replay counterpart and are listed explicitly in the comparison JSON;
these are additional coverage, not numeric mismatches.

Differences left for Pat to adjudicate, with neither side corrected:

- **R3 qualification:** the frozen verifier includes lithium and carbon
  native/random frontiers and qualifies each as `boundary`. The harness
  excludes LI and C with `no native evidence`. The raw native top-1s are
  17623 and 236780 and are unchanged by the random edit. C6 remains
  NOT_EVALUATED on both sides, so this evidence-coverage difference does
  not change a control status or the verdict.
- **R4 form:** Codex classified the observed donor replay at the saved
  native answer boundary as response-time triggered, using the native
  proposal, and not applied at every forward. The harness classifies the
  subject-substitution rule as static: the respective flags are false,
  false and true. The other form properties agree. C0 remains FAIL on
  both sides because donor inference, subject substitution and absence
  of a single stored artifact already disqualify this case. This is an
  interpretation of candidate form versus measurement procedure, not a
  numerical discrepancy.
- **R1 donor-ceiling coverage:** Codex retains the archived symbol donor
  observation, top-1 He, as a nonblocking C5 observation; the number donor
  is explicitly marked outside C5's efficacy domain. The harness C5
  detail says `no donor evidence`. Both C5 statuses are NOT_EVALUATED.
  No harness ceiling number is present to compare; this is reported as
  an evidence-coverage difference.

The complete mappings, all unequal fields, all unmatched numeric items,
the six harness report SHA-256 values and hashes of the three frozen
files are in
[`research-data/edit-cert-replay-blackbox-comparison-codex.json`](research-data/edit-cert-replay-blackbox-comparison-codex.json).
The one-off comparison generator is archived alongside the black-box
reports as `edit-cert-replay-blackbox-codex-20261026/compare-codex.py`,
with its hash in that JSON. It only reads the frozen output and emitted
harness reports; it does not import harness code or run inference.

The original three frozen files still match their commit byte for byte.
The uncommitted L14 partition runner's SHA-256 remains
`a77eff37a7c801c91621256510ec759cca2c3605537a740873dbb50ba9ebe7b5`.
No model passes, harness corrections, frozen-result revisions or pushes
were performed. This comparison verifies the five archived replays; it
does not certify the harness on other inputs, demonstrate persistence or
establish an edit of a fact.
