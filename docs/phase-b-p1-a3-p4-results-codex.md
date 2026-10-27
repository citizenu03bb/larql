# P1-4 A3: independent Codex comparison — 2026-10-27

**All five facts agree under the unchanged A1/A3 thresholds.** All ten
saved affine constraints pass. No gate ambiguity or recorded hash mismatch
was found. P1-5 may therefore certify the **five existing Codex edits**,
within its approved maximum of **4334 forwards**. This agreement establishes
candidate-construction replication, not edit certification.

## Chronology and independence

Pat's briefs 44/45 report A3 approval and completion of Claude's five new
solves. Codex committed its exact conventions at **60482963**, after the
solves existed but before reading their factor/status artifacts or any A3
comparison. The document and receipt remain unchanged. This declaration
was not pre-execution, and that chronology is explicit.

Codex then sealed `solves-claude-a3/completion.json` with SHA-256
`725a06742a5c22e644ee980ebd560516fbc7f4f9c7d9e5d0ef8405845f7a504d`,
matching the communicated prefix/suffix. Only schema, filenames and input
metadata were inspected before writing the I/O wrapper. The new wrapper
was committed at **dd12dc99 before its first execution**; its threshold
math is the unchanged module committed at **c713a033** before the original
P1-4. No peer solver, comparator, verifier code or A3 comparison report was
read. This is Codex's independent score, not a comparison to Claude's score.

The output was created once in a new file. No factor normalization, sign
change, threshold change, candidate repair or new solve occurred. This
stage uses **zero model forwards**. The original ten candidates and old
0/5 gate replay unchanged.

## Frozen gate

The gate requires directed cosine U and V at least 0.95 and update norms
within 10%. Norms are `||U|| * ||V||` at scale one. The percentage below is
the largest relative difference among the three denominator conventions
already disclosed before A3; all three give the same decision.

| Fact | cosine U | cosine V | Codex norm | Claude A3 norm | Maximum norm difference | Gate |
|---|---:|---:|---:|---:|---:|---|
| Bmw | 0.999999490 | 1.000000000 | 26.935550 | 26.931605 | 0.014649% | AGREE |
| Germany | 0.999999684 | 1.000000000 | 8.207782 | 8.206066 | 0.020917% | AGREE |
| Gretzky | 0.999999431 | 1.000000000 | 9.365285 | 9.367602 | 0.024746% | AGREE |
| Nintendo | 0.999990781 | 1.000000000 | 1332.141870 | 1332.097936 | 0.003298% | AGREE |
| Shakespeare | 0.999995134 | 1.000000000 | 8.119579 | 8.117352 | 0.027437% | AGREE |

The minimum U cosine is **0.999990781**. The largest relative norm difference
is **0.027438%**, far below 10%. All V directions are effectively identical.
There are **5 agreements, 0 disagreements, 0 ambiguous cases**. A3's
3-disagreement closure is not triggered.

## Input identity, integrity and algebra

- All five A3 completion entries are complete with **462 sequences / 20
  updates** each: **2310 forwards / 100 updates**, without a reported retry.
- All five status hashes match the sealed completion. All five backend
  content hashes match both status and completion. U/V witnesses exactly
  match served arrays. All previously recorded own/original hashes verify.
- The shared model, C17, templates, hyperparameters, fixed update exception
  and subject positions match. The **21 context keys are bitwise identical
  for every fact**, as are the separate clean native output captures.
- Maximum absolute affine error is **4.872e-7 for Codex**, **1.005e-6 for
  Claude A3**, below the previously declared float32 tolerance.

Claude A3 saves target_init and delta rather than a standalone v_star.
Before scoring, the wrapper fixed its primary reconstruction as their
float64 sum, following the declared float64 U construction, with the
separate cur_output as affine base. A float32 sum is also reported solely
as a serialization diagnostic: it passes as well, with maximum absolute
error 1.421e-6. It does not alter the factor gate or choose a favourable
algebra convention after seeing its result.

The completion index pins status hashes and edit content, rather than
individual hashes for every auxiliary witness. The independent output now
records all **60 observed A3 file hashes**; it does not claim those witness
hashes were individually sealed in advance. The
[full comparison JSON](research-data/phase-b-p1-a3-pair-xcheck-codex.json)
retains every metric, input hash and witness error.

## Interpretation of the declared numerical differences

Claude's batched target_init differs from its own singleton cur_output:
the L2 difference ranges from **0.005731 to 0.050775** across facts. That
capture distinction was declared before this comparison. Codex's base is
its singleton capture; the two sides' singleton bases are bitwise equal.

The observed residual factor differences are compatible with the already
declared capture, rounding and KL-baseline conventions. Their separate
causal contributions have not been established. No ablation or replacement
comparison is proposed. In particular, Nintendo's norm remains about
1332 on both sides, consistent with the previously declared small clean-key
denominator; its much larger norm is not evidence of certification.

## P1-5 handoff and limits

The [certification input manifest](research-data/phase-b-p1-a3-certification-handoff-codex.json)
pins each unchanged **Codex** edit path/content hash, solver manifest, and
native-qualified specification, with these A3 caps:

| Fact | Bench cap | D1 probe cap | Total cap |
|---|---:|---:|---:|
| BMW | 832 | 54 | 886 |
| Germany | 784 | 54 | 838 |
| Gretzky | 784 | 54 | 838 |
| Nintendo | 832 | 54 | 886 |
| Shakespeare | 832 | 54 | 886 |
| Total | 4064 | 270 | **4334** |

Claude runs certification; Codex independently cross-checks its archived
results. Each actual plan must be committed before its first P1-5 forward,
under unchanged controls, stop/negation/C7 readings, fresh dry runs, the
shared flock and bounded services. P1-5 services keep their **400-forward**
cap; the 500/25 exception belonged only to solves. A failed service stops
its step without a retry. The manifest prepares inputs and launches no
service.

Actual spending since P1 v2 is now **9030** forwards, with at most **4334**
remaining for P1-5 under the approved **13364** total. Including reported
closed-v1 qualification adds 500 to both ledger and ceiling. No forwards
were consumed by this comparison or handoff.

This comparison establishes **no CERTIFIED verdict**. The existing W1 exclusions
for BMW/Gretzky cap full certification at **3/5** and the best reachable
reading remains partial. An agreement or later GPT-2 result alone cannot
attribute all prior E4B failures solely to architecture or quantization.
No blind-set access, downloaded model, new solve or push occurred.

The five inherited comparison-math and integrity tests passed before the
new comparator was committed and executed. Handoff hashes and the sum of
all per-fact caps were read back and checked without re-scoring the gate.
