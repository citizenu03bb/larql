# Research branches: natural fact editing in Gemma 4

*Draft index, 2026-10-26. Not yet pushed: to be published with the consolidation push, after a secret scan. Commit references marked "pending push" are local until then.*

This fork of LARQL holds an exploratory research series on one question: **can a fact the model already knows be edited selectively, persistently, and robustly across phrasings?** The series uses one native fact in Gemma 4 E4B, hydrogen's chemical symbol, with a deliberate counterfactual H → He.

**Where it stands.** The mechanistic series is closed and written up. An activation delta keyed on the entity is not a selective, persistent fact edit in that form. The next step is a method-agnostic certification harness, built on this branch. After that comes a weight-editing method early in the network, judged by the harness.

## Start here

- **Write-up:** [`docs/activation-patching-followup.md`](https://github.com/citizenu03bb/larql/blob/research/nf4-h1-name-form/docs/activation-patching-followup.md) on `research/nf4-h1-name-form`. It covers the K/V-sharing write deadline, the symbol/atomic-number asymmetry, why the static form loses its selectivity, and what is not claimed.

## Branches

| Branch | Agent | What it holds |
|---|---|---|
| `research/exploratory-natural-fact-edit` | Codex | The series from the start: Q4K experiments and their French journal and status; the NF4 local bench and its K/V-boundary check; the asymmetry reproduction; the PLE factorial (H2); the timing test (H3); the independent cross-check of the certification replay. Public up to `20eb6635`; the PLE, H3 and cross-check commits are pending push |
| `research/nf4-h1-name-form` | Claude Code | The name-form test (H1) and its post-hoc matrix; Phase 3 (static patch) and Phase 3b (row split); cross-checks of the PLE and H3 runs; the English write-up. Branched from the Codex branch at `20eb6635`. Public at `c298ca79` |
| `research/edit-certification` | Claude Code | The edit certification harness (`tools/edit_cert/`): design, controls C0–C8, verdict rule, model-free tests, replay of five archived cases. Branched from `research/nf4-h1-name-form` at `c298ca79`. Pending push |

## Reports, in reading order

| Step | Report | Branch | Ran it | Cross-checked by |
|---|---|---|---|---|
| NF4 bench qualification, K/V boundary on E2B | `docs/nf4-local-bench-results-20261026.md` | Codex | Codex | — |
| Asymmetry reproduced on NF4 | `docs/nf4-asymmetry-results-20261026.md` | Codex | Codex | — |
| H1, name form | `docs/nf4-h1-name-form-results-20261026.md` | H1 | Claude Code | — |
| H2, PLE factorial | `docs/nf4-ple-factorial-results-20261026.md` | Codex | Codex | Claude Code (`docs/research-data/*crosscheck-claude.json`) |
| H3, timing | `docs/nf4-h3-timing-results-20261026.md` | Codex | Codex | Claude Code |
| Phase 3, static patch | `docs/nf4-phase3-static-results-20261026.md` | H1 | Claude Code | — |
| Phase 3b, row split | `docs/nf4-phase3b-rows-results-20261026.md` | H1 | Claude Code | — |
| Certification harness and replay | `tools/edit_cert/README.md` | edit-certification | Claude Code | Codex (`docs/edit-cert-replay-xcheck-codex.md`, frozen at `5b85fba8` before running the harness; comparisons `64961ae8`, `05768fb6`, `18bef9ec`; R4 form reading `1aae8a1a`) |

Every series froze its protocol before the first forward, in a commit. Each has:
- a model-free checker;
- compact evidence JSON;
- a local archive with a SHA-256 manifest.

Unless a cross-check is listed, the checker was written by the agent that ran the series.

## What is not claimed

- No knowledge edit, no persistent edit and no generalisation across facts. One fact, a handful of phrasings, one greedy trial per case.
- No entity–relation mechanism is established.
- The certification harness has certified **no real edit**. Its only `CERTIFIED` verdict comes from a synthetic self-test. The five archived cases are all refused.

## Who did what

The work was designed, coded, run and verified by AI coding agents, Codex and Claude Code, under the direction of Patrick Kappeler. He set the mandates and budgets, decided at forks, and convened audits and model panels.
