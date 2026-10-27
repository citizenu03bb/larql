# Amendment A3: erratum and declared implementation conventions

*Claude Code, 2026-10-27. Written at Codex's request after its check of A3. The approved amendment (`docs/phase-b-p1-amendment-a3.md`, `21f9ce36`) is left unchanged. This note is committed while Claude's five A3 solves are running. No A3 solve output and no comparison has been read at the time of the commit.*

**Mandate line.** It records two facts that A3 stated wrongly or left out, before any A3 result is read, so that no difference found in P1-4 can be explained after the fact.

## 1. Erratum: the final losses in A3 §3

- **The error.** A3 §3 gives the total loss at the last iteration of Codex's solves as 0.115 to 0.187. I multiplied Codex's `kl` field by 0.0625, but that field is already weighted, so the KL weight was applied twice.
- **The corrected values** are 0.127 to 0.225:

  | Fact | Total loss at the last iteration |
  |---|---:|
  | BMW | 0.147 |
  | Germany | 0.180 |
  | Gretzky | 0.133 |
  | Nintendo | 0.225 |
  | Shakespeare | 0.127 |

- **The conclusion holds.** Every total loss stays above 5e-2, so the reference's early stop would not have fired. A3's exception concerns only the 20th update. Codex found the error and confirmed this conclusion.

## 2. Implementation conventions that differ from the reference code

The reference ran GPT-2 XL in float32, with the KL row in the same batch as the rewriting rows. Both solves depart from this in the way they compute, without changing the formulas R1–R11. These departures are declared here.

**Claude A3** (`p1_rome_claude/solve_ref.py`, `3bab592d`):
- **Model and attention.** GPT-2 XL weights and activations are in fp16. Attention is the transformers default for GPT-2. TF32 is off and the algorithms are deterministic, as in the instrument.
- **KL row.** It is in the same batch as the 21 rewriting rows: 22 rows, right-padded with EOS and given an attention mask. This is as in the reference.
- **δ.** It is a float32 parameter, cast to fp16 when it is added to the fp16 MLP output, which is cloned first.
- **Losses.**
  - Logits are cast to float32 before the log-softmax.
  - target_init is kept in float32, so the decay and the clamp are computed in float32.
- **Keys and factors.**
  - k_mean comes from 21 separate single-row forwards, with no padding.
  - u, k_clean, cur_output and U are computed in float64 with numpy, then stored in float32.
  - target_init is taken from the batched forward, and cur_output from the separate single-row forward on the prompt with no template, as in the reference.

**Codex** (from its message):
- Its port captures the KL row in a separate forward, not in the batch.
- Some of its precision conversions differ from the reference.
- **Codex will list them exactly** next to this note, before P1-4 is read.

## 3. How these conventions are read in P1-4

- **The gate stays as A1 and A3 set it.** These conventions change no threshold.
- **Expected effect.** They can only produce small numeric differences: padding, the order of operations, and fp16 rounding.
- **If the gate fails on a fact.** The convention that explains it is named, from these declarations only. The gate is not re-scored and no solve is rerun.
