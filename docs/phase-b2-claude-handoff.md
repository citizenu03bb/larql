# B2 handoff: independent algebra checks and dev certifications

For Claude Code, under Pat's approved `phase-b-rome-b1-b2-v1` protocol.
Codex owns the method; Claude owns all certifications. No blind test set
may be read or released at this stage.

Run root: `~/Projects/larql-research/phase-b-rome-20261026/`.
The model-free `certification-handoff-latest.json` inventory lists the 15
fixed slots (five public facts, L4/L8/L12), with explicit pending/failed/
complete states. Only complete edits have certification commands. Immutable
handoff snapshots are also written as solves complete. The inventory is
not a verdict, and its C1/C2 fields remain null.

B1 completed 200 real forwards and produced the registered covariances.
Its receipt is `b1/status.json`, with C4/C8/C12 hashes; the checkpoint's
full SHA-256 is in `preflight.json`. Each complete edit directory includes
backend.json, U0/V0, k_star, v_star, w_k_star, C, context keys/native values,
seen phrases, optimizer trace and source hashes. The whole edit is a
runtime low-rank weight adapter; no answer-time trigger is installed.

Before certifying an edit, independently recompute its algebra without
importing Codex's solver or verifier:

```text
k* = mean(context_keys, float64 accumulation), cast to float32
q = solve(C, k*)
U = (v* - Wk*)[:, None]
V = (q / (k*.T @ q))[:, None]
```

Compare U0/V0 numerically and check the registered float32 mean-key
constraint at rtol/atol 2e-5. Wk* is explicitly saved so this check needs
no model. Report any differences; do not silently repair or rescale them.
Check saved C against B1 and artifact/metadata hashes against the service
receipt. This algebra check does not replicate value optimization or
establish fact editing.

Plan each complete case using the inventory's command, in the harness
worktree, with `--instrument nf4-e2b --cap 400` and the matching original
public dev spec. S1 is seen; all training contexts and the essence prompt
are in backend provenance. Keep every existing native qualification,
efficacy, attribute/order, answer-only, random, off-target and reload rule.
At 24 new tokens, a full dev plan has 832 worst-case forwards in four
services (312, 400, 24, 96). No sample replacement or hyperparameter change.

Every model service must hold the same shared heavy-model flock through
systemd-run --wait, with 22G/24G/8G and 1800 seconds. Codex's solve services
use that lock too; let the holder finish. Planning and algebra checks need
no model and can run while the GPU is occupied.

Please return an index of the 15 independent verdict files and raw service
records, with hashes, actual counters and completeness. Preserve underlying
generated IDs/text and boundary artifacts needed for a later model-free
cross-check. Report ineligible or failed cases; do not replace them.

Codex will select the largest count of C1+C2 PASS among the original five
facts, ties to the earlier layer, only after those independent results.
Below 2/5 at every layer: stop. Above the gate: create the five identity-C
ablations at the selected layer, with no extra optimization, for Claude's
independent certification. Development is not a test result. B3 unblinding
requires a separate method-freeze commit and release, never this handoff.
