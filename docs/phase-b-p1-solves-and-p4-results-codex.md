# P1 independent Codex solves and P1-4 comparison, 2026-10-27

All five approved Codex solves completed: **2310 sequence forwards and 100
Adam updates**, exactly 462/20 per service, no failed solve and no retry.
Each fresh model service followed a fresh synthetic rehearsal, held the
shared flock, and used the fixed 500/25, 22G/24G/8G and 1800-second bounds.
The covariance attempt and A2 together used 1472 forwards: Codex's actual
P1-2 plus solves total is **3782 forwards**, including the discarded A1.
No new downloads, blind-set access or push occurred.

The [delivery manifest](research-data/phase-b-p1-solves-codex.json) pins
all twelve payload hashes for each solve, source hashes, launch receipts,
rehearsals, counters, model identity and shared input hashes. Raw arrays
remain in `~/Projects/larql-research/phase-b-p1-v2-20261027/solves-codex/`.

## Independence and integrity

Before our fifth solve completed, the only peer artifact read was
`solves-claude/completion.json`. Its bytes matched the snapshot sealed
before our first solve (SHA-256
`07d7f2584720032b4c3cb57a5821f8c52c771e5a3d740c291ffc958a3593c0e4`).
The comparison math and interpretation were committed before peer-factor
access at c713a033 and 4c36cdeb. P1-4 began only after both complete
five-fact indexes existed. No peer source was read, including in P1-4.

The model, C17 and shared template identities match. All sixty own payload
hashes, all five own manifests, all five peer status hashes, and all ten
edit content hashes verify. Peer exported U/V witnesses exactly match
their served backend arrays. Peer witness npy files have no separate hashes
in the previously sealed completion index: their hashes are recorded now;
that index previously pins the statuses and edit content. No missing or
mismatched recorded hash was found.

## Frozen comparison result: 0/5 agree

Norms below are the rank-one update Frobenius norms `||U|| * ||V||` at
scale 1. Directed factors were neither sign-flipped nor rescaled.

| Fact | cosine U | cosine V | Codex norm | Claude norm | max/min | Agreement |
|---|---:|---:|---:|---:|---:|---|
| Bmw | 0.658361 | 1.000000 | 26.935550 | 3.805095 | 7.079 | FAIL |
| Germany | 0.732127 | 1.000000 | 8.207782 | 5.009946 | 1.638 | FAIL |
| Gretzky | 0.591068 | 1.000000 | 9.365285 | 2.553555 | 3.668 | FAIL |
| Nintendo | 0.711579 | 1.000000 | 1332.141870 | 3.142444 | 423.919 | FAIL |
| Shakespeare | 0.751789 | 1.000000 | 8.119579 | 3.401131 | 2.387 | FAIL |

Every U cosine is below 0.95 and every norm difference is above 10%, under
all three disclosed denominator conventions. All V cosines are effectively
one. **Five disagreements out of five block P1-5 under A1.** None of these
edits has been certified; no certification or replacement solve is launched.

Both candidates' saved affine rank-one constraints pass for every fact:
maximum absolute error 4.872e-7 for Codex and 1.242e-7 for Claude, below the
predeclared float32 tolerance. This checks each candidate against its own
saved key, target and affine base, not against a newly chosen constraint.
Claude's reported pre-export constraint errors need not equal errors after
float32 serialization; both are preserved.

The [full model-free comparison](research-data/phase-b-p1-p4-pair-xcheck-codex.json)
and `tools/rome_solver/xcheck_p1_pairs.py` record every number and hash.
The frozen math tests and integrity tests pass; P1-4 uses zero model passes.

## Differences visible in saved artifacts

The 21 captured context keys are **bitwise identical for all five facts**.
Thus a different context-key capture is not the observed source of this
disagreement. The saved metadata and witnesses expose different solves:

- Codex optimizes an additive delta at each context's native MLP output,
  with `0.5 * ||delta|| / ||clean_native_value||^2`. Claude's statuses
  explicitly declare replacement of the c_proj output by z and a squared
  relative L2 penalty. The loss traces consequently have different scales.
- Codex's affine constraint uses the clean context key; Claude's saved key
  matches the mean of the 21 keys (L2 difference 2.5e-7 to 4.1e-7), and
  differs from the clean key. Both input-factor directions agree, since
  both use the same covariance-weighted mean direction.
- Nintendo illustrates the constraint sensitivity: Codex's unit V has dot
  product 0.1131207 with the clean key and 6.9051901 with the mean key.
  The clean constraint therefore requires a much larger output factor.
  This is a measured property, not grounds to replace the constraint.

These differences are directly established from outputs, without reading
peer code. Their separate causal contributions to the final discrepancy
have not been isolated. The independent rank-one algebra passes do not
establish that the optimization objectives are equivalent.

Recommendation: reconcile the mathematical specification against published
ROME without a model, record which objective, hook and constraint are
intended, and freeze an explicit amendment before considering any new
solve. Preserve these ten candidates and this failed agreement gate. Do
not select or normalize a candidate after seeing its results.

The pre-existing native coverage limitation also remains: BMW and Gretzky
lack a qualified mandatory answer-only witness, so the current five-fact
set has an upper bound of three fully certified edits. Nothing in P1-4
changes the native qualification or justifies a claim about persistent
fact editing, or about the cause of the prior E4B results.
