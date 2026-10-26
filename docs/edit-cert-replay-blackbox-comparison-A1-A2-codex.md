# Second black-box replay: amendments A1/A2

Mandate: `career-strategy/31-codex-relance-comparaison-edit-cert.md`.
Harness commit: `620a50b081ec6ad723121fd2591507d610def5e5`, on a clean
`research/edit-certification` worktree. Independent freeze:
`5b85fba8f6a147aeae88b30ba813c8dba3081b9c`. First comparison:
`64961ae8`. No model passes, pushes or changes to either preserved result.

The black-box replay succeeds, but **the original comparison generator,
left unchanged as requested, fails on a newly added quantity**. The field
agreement below is a separately labelled supplementary inspection; it is
not a claim that the unchanged-generator procedure completed successfully.

The command was run from the harness's `tools/edit_cert` directory:

```sh
python3 -m edit_cert replay \
  --out "$HOME/Projects/larql-research/exploratory-natural-fact-edit/edit-cert-replay-blackbox-codex-A1-A2-620a50b0-20261026"
```

Exit code 0; `exit_criterion_met: true`. Reproduction counts are R1 15/15,
R2 13/13, R3 7/7, R4 2/2 and R5 8/8: 45 items in total. The amendment
passages in design §11 and `AMENDMENTS` were read only after the original
freeze; the numerical inspection reads emitted reports, not adapter code.

The same generator, SHA-256
`2593968f3866fbf5001446f5879ad22310c8cbca24f1eb891c87d5214e3adeb6`,
was copied byte for byte into the new archive folder and executed. Its
fixed input/output paths were redirected by an invocation wrapper; no
JSON content, mapping or generator source was changed. Writes to the first
comparison were prevented. Execution stopped with:

```text
KeyError: 'symbol_donor_argmax'
```

The original generator expects R1 replay quantities to be flat keys in
`cases.R1.numbers`. The frozen R1 donor top-1 is instead already present
at `cases.R1.donor_ceiling_observations.symbol.top1`. The generator has
no mapping for that new replay item. It produced no second comparison
output. Its source was not repaired, and neither the harness nor the freeze
was changed to accommodate it. Strict completion of mandate step 2 is
therefore blocked by this comparator limitation.

A separate diagnostic inspected every reported item against existing
fields of the frozen JSON. It reuses the first comparison's 40 mappings,
checks their values against the freeze, and explicitly records five new
field locations. This additional inspection gives:

| Check | Agreement | Differences |
|---|---:|---:|
| Verdict, case report and replay report | 5/5 | 0 |
| C0–C8 status | 45/45 | 0 |
| Numeric/boolean replay items | 45/45 | 0 |
| Qualification level | 18/18 | 0 |
| C3 before/after groups | 10/10 | 0 |
| C0 form properties | 32/35 | 3, all unchanged in R4 |

All verdicts are REFUSED. All 45 items also exactly equal the archived
references reported by the harness. This diagnostic does not rely on the
harness's `expected` verdict table or its self-reported reproduction flags.

The five new replay items are already evidenced in the original freeze:

| Case / quantity | Value | Frozen location within the case |
|---|---:|---|
| R1 symbol donor top-1 | 2209 (He) | `donor_ceiling_observations.symbol.top1` |
| R3 lithium native top-1 | 17623 | lithium observation, `native_top1` |
| R3 lithium random top-1 | 17623 | `numbers.lithium_random_argmax` |
| R3 carbon native top-1 | 236780 | carbon observation, `native_top1` |
| R3 carbon random top-1 | 236780 | `numbers.carbon_random_argmax` |

A1 closes both R3 qualification differences: LI and C are now `boundary`,
matching the freeze. C6 stays NOT_EVALUATED. A2 closes the R1 donor
coverage difference: C5 reports `observed_on_seen: {"SYM": [2209]}` and
stays NOT_EVALUATED. No efficacy PASS is inferred from that seen boundary.

The three R4 form differences remain exactly as before, for Pat to
adjudicate:

| Property | Frozen Codex reading | Harness reading |
|---|---|---|
| Response-time trigger | true | false |
| Uses native proposal | true | false |
| Applied at every forward | false | true |

Both sides still give R4 C0 FAIL and C3 FAIL. No verdict, control-status,
qualification or numeric discrepancy was found in the supplementary
inspection. These checks validate this replay composition, not general
harness correctness, persistent edits or an edit of a fact.

The machine-readable diagnostic, including the unsuccessful unchanged
generator attempt, exact field mappings and source hashes, is
[`research-data/edit-cert-replay-blackbox-comparison-A1-A2-codex.json`](research-data/edit-cert-replay-blackbox-comparison-A1-A2-codex.json).
The new archive folder also contains `run-original-comparison.py`, the
byte-identical `compare-codex.py`, the exception log, the explicitly
separate `supplementary-inspection.py`, and `preservation-receipt.json`.
The receipt verifies the three frozen files, both first-comparison files,
the six first harness JSON reports and the original generator unchanged.
The unrelated uncommitted L14 runner also retains its previous SHA-256.
