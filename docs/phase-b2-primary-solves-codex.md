# B2 primary solves complete; independent certification pending

All 15 registered primary solves completed: five original public dev facts
at L4, L8 and L12, in the frozen lexical/layer order. Every service used
130 forwards and 25 Adam updates. There were no failed solves, retries,
replacement facts or changes to the frozen method/protocol. B2 solving
used 1,950 forwards and 375 updates; with B1, Codex's method execution used
2,150 real forwards. Certification forwards are accounted separately by
Claude Code.

Each model service had a fresh same-wrapper synthetic rehearsal and held
the shared heavy-model flock through bounded systemd execution. Completion
receipts, all exported array hashes, solver manifest hashes and the saved
covariances were rechecked by the model-free inventory. All 15 edits were
ready, with zero pending or failed slots. Raw artifacts remain under
`~/Projects/larql-research/phase-b-rome-20261026/`; the compact receipt is
[phase-b2-primary-solves-codex.json](research-data/phase-b2-primary-solves-codex.json).

No loss curve or algebraic constraint is treated as evidence of fact
editing. Claude Code independently recomputes the algebra and certifies
each edit. His incremental raw index is
`~/Projects/larql-research/phase-b-certification-20261026/index.json`.
Its state `certified` means the certification process finished, not that
the verdict is necessarily `CERTIFIED`. Only `complete: true` after all
15 cases allows the registered layer decision.

## Independent gate checker frozen before comparison

`tools/rome_solver/xcheck_b2_gate.py` independently recomputes C1/C2 from
raw service generations, checks their hashes against the index, matches
the aggregated evidence to those raw records, and verifies actual forward
counts. It imports no harness judging logic and does not independently
certify C0 or C3--C8. Six model-free edge-case tests pass. The initial
output, [phase-b2-gate-xcheck-codex-prefreeze.json](research-data/phase-b2-gate-xcheck-codex-prefreeze.json),
covers the five completed peer cases available at the snapshot. Expected
peer C1/C2 values and verdicts have not been inspected before this freeze.
Later complete runs will reuse the committed checker without tuning.

Interpretations are explicit in the output. Native cap hits are excluded;
an edited cap hit never passes. An explicit rejected/old value or swap
marker is a counterexample even in a truncated edited answer; otherwise
that truncated answer is unreadable. A completed answer missing the
accepted value fails. C1 requires the public spec's minimum unseen count
and at least one qualified positive in each subject order. Seen questions
can fail but never provide positive coverage. C2 requires preservation in
both orders. Differences from the independent harness are reported, not
silently corrected on either side.

The layer gate remains the count of **reported C1 and C2 both PASS** among
the original five facts, ties to the earlier layer. Below 2/5 everywhere,
stop. If it clears, export the five registered identity-C ablations with
no new optimization and obtain independent certifications. No blind test
set was opened, no identity rescue or layer selection was made, and no
push was performed.
