# Question Audit

Status: checklist passed by Claude Code (synthesizer, non-contributor) before launch, 2026-10-27; written to this file just after launch, unchanged in substance. Not audited by a second party.

## Checklist

- [x] Carrier terms checked against authoritative project sources: control names and rules (C0–C7, asymmetry) from `docs/edit-certification-design.md` and `controls.py`; ROME hyper-parameters from Codex's `docs/phase-b-b1-b2-protocol.md`; gate rule, stop rule and pre-written readings from `docs/phase-b-decision.md` §4; results from `docs/phase-b2-v2-certification-results.md`.
- [x] Implicit assumptions checked: one-week limit (from 2026-10-26), NF4 reported as NF4, test set sealed and unopened, second method pre-designated.
- [x] Prompt allows reframing: item (d) asks for implicit assumptions in the wording, the options and the bench itself; option (D) is open.
- [x] Imprecise terms: "C3" defined as the answer-only mention control as implemented in v2; "fuite" given as measured examples.
- [x] Epistemic status: results labelled measured, same-agent recomputation, cross-check pending; bleed causes explicitly labelled untested hypotheses; v2 amendments stated as written after v1 data, before v2 forwards.
- [x] Double-interpretation phrasing: options stated as disjoint propositions; continuation after the stop rule stated as a new decision.

## Notes

- **Framing bias disclosed.** The framer (Claude Code) had already recommended option A to Pat before writing the question. The recommendation is not in the question, options are listed symmetrically, and option B (also Claude's earlier suggestion) is stated without advocacy. The rule "never framed after giving one's opinion" is therefore not met to the letter; this is recorded rather than hidden.
- **Lineup.** Third run of the same question, unchanged, requested by Pat: gemini-4-argon, kimi (moonshotai/kimi-k3.1-pro), qwen (qwen/qwen3.8-max-prime), grok (x-ai/grok-4.7), minimax (minimax/minimax-m3). Codex and Opus still excluded as authors.
- **Framing defects found after launch (all three runs).** (1) The question defined C3 as the answer-only mention only; in the harness C3 is "mentions before and after the relation": it also requires the new value in both subject orders, so it fails whenever C1 fails, and the answer-only substitution is one of its cases. (2) The question omitted that Claude's model-free algebra cross-check (W′k* = v* within 2e-5, artifact hashes, C against B1) passed on all 15 solves. Neither defect changes a recommendation; both are reported in the global synthesis.
