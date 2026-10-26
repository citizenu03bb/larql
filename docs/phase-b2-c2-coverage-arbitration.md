# C2 native-qualified coverage: Pat's post-comparison arbitration

The independent C1/C2 checker and its first output were frozen at
`1d6af9e8`, before reading peer statuses. The first comparison used five
completed cases (ten controls) and found two C2 differences, with all
checked file hashes and counters matching. Neither side's output or code
was modified to remove a difference.

| Case | Harness C2 | Codex C2 |
|---|---|---|
| Beethoven L8 | PASS | NOT_EVALUATED |
| Beethoven L12 | PASS | NOT_EVALUATED |

Both sides exclude A1 (subject after relation): the native answer contains
the expected disability, but is truncated at 24 tokens without EOS.
Only A2 (subject before relation, nine symphonies) is qualified and kept.
The harness reports `1 attributes kept` and passes C2 on that one order.
The independent checker requires both orders, following the design's
section 3 wording, `in both orders`, and returns NOT_EVALUATED.

Pat explicitly chose **“Exiger les deux ordres : NOT_EVALUATED”** after
being presented this exact discrepancy. This is a post-comparison
arbitration for the B2 decision, not a pre-registered result. A separate
`apply_b2_c2_arbitration.py` receipt records the effect: a reported C2 PASS
without native-qualified attribute coverage in both orders becomes
NOT_EVALUATED for the development gate. A reported FAIL remains FAIL.
All other differences remain visible and need separate arbitration before
a positive continuation decision.

The original five-fact denominator, maximum-count rule, earliest-layer
tie break and 2/5 threshold remain unchanged. The receipt displays both
the reported statuses and the decision statuses. It does not rewrite
Claude's verdicts, the frozen checker, its first output, or the original
B1/B2 protocol. It uses no model passes. The raw first comparison is
preserved in `phase-b-rome-20261026/b2-gate-xcheck-first-comparison.json`.
