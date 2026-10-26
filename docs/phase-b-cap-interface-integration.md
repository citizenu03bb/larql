# Phase B service-cap interface integrated

Following `career-strategy/34-codex-reponse-interface-plafond.md`, this
integration merges Claude Code's `28aef666dcdfe6530dcaad4c9fc6f2faa65286d1` into
`research/phase-b-rome`. A merge preserves the B0 commit `70f0cd6c` and
its frozen protocol. All 28 B0 files remain byte-identical to that commit.
The historical interface request and B0 cost sheet remain unchanged.

The request in `phase-b-harness-interface-request.md` is now resolved:
public prepare and plan CLI accept an explicit approved cap of 400, retain
24 by default, and hash cap/authority into the plan. No method-side or
certification-rule change was made by Codex.

Validation after integration, without a pretrained model:

- Harness pytest: **58 passed**, 6 torch skips covered separately.
- Existing torch backend tests: **6 passed**.
- ROME CPU synthetic tests: **16 passed**.
- Public CLI planning on the Eiffel dev spec, with a temporary zero
  adapter of the declared E2B dimensions: cap 400 gives **312 + 400 + 24 +
  96 = 832** forwards. The plan reloads with its hash valid and records Q1
  authority. This is planning only; no service was executed.
- Omitting `--cap` retains 24 and null approval. Cap 100 and cap 400 on
  `larql-q4k` both fail before creating the run directory.

The [receipt](research-data/phase-b-cap-interface-integration.json) records
the CLI results and the hashes of every preserved B0 file. Harness files
match `28aef666` exactly. No push, model load or pretrained forward occurred.
The blind directory was not accessed.

This integration resolves the interface dependency; it does not authorize
B1/B2. Those stages still await Pat's explicit approval of the committed
`phase-b-rome-b1-b2-v1` protocol. Native cache and multi-fact certification
remain with Claude Code for B4.
