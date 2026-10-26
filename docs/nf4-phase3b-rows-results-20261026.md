# NF4 Phase 3b row-split diagnostic — 2026-10-26

**Outcome: MENTION holds, STEP does not.** The number effect that stopped Phase 3 comes from the answer's own "Hydrogen" mention, which has already read the question, and not from the prompt's "hydrogen". Even a single application on that row gives " three", not " two".

## Protocol

The protocol was frozen in `scripts/run_nf4_phase3b_rows.py` (commit `2206b4b7`) before any forward. It was motivated by the Phase 3 stop, a post-hoc origin stated in the protocol.
- **Patch.** The same archived L21 vector, applied post-block at L21 with PLE native, on the two archived resumed boundaries.
- **Conditions.** Native; row 5 only at g=1 (the prompt's "hydrogen", before the relation); row 30 only at g=1 (the answer's "Hydrogen"); both rows at g=0.5.
- **Budget.** 8 forwards, one service, 24.4 s, 8.1G peak, no swap, shared lock held.
- **Checks.**
  - The native replays and the row-5 number condition are bit-identical to the archived logits.
  - The same-author checker returns `verified_complete`: hashes, margins, exact requested vectors, applied increments and readings.

## Results

Each cell gives the top-1 token and the increment of the target − native margin.

| Condition | Symbol: top-1, He−H increment | Number: top-1, two−one increment |
|---|---|---|
| native | H, 0 | one, 0 |
| row 5 only (prompt), g=1 | H, +19.73 | one, **+0.05** |
| row 30 only (answer), g=1 | **He**, +45.01 | **three**, **+15.58** |
| rows 5+30, g=0.5 | He, +41.80 | three, +16.54 |

## Reading

- **MENTION holds.** On the number boundary, the row-30-only increment is ≥10 and the row-5-only increment is <2.
- **STEP fails.** " two" is not top-1 in any condition. The post-hoc idea of a single helium-ward step along element order is **not supported**: one application on row 30 already gives " three".

What this adds to the series:
- **The L21 selectivity is a property of where the entity sits.** Placed before the relation ("Give hydrogen's …"), the subject row at L21 changes the symbol and leaves the number. A mention that has already read the question carries the delta to both read-outs, and pushes the number to a value that is neither hydrogen's nor helium's.
- **A persistent edit keyed on the entity cannot be selective in this form.** It fires on every mention, including relation-aware ones, so it cannot keep the selectivity seen at row 5.
- Natural questions such as "What is the atomic number of hydrogen?" place the entity after the relation. There, even a prompt-only trigger would hit a relation-aware row.

## Limits

- One NF4 model, two resumed boundaries, one trial per condition, a single gain per condition.
- Why the value is " three" is not explained; any account is speculative.
- No free generation, no Q4K/LARQL confirmation.
- No weights, vindex or persistent patch written.
