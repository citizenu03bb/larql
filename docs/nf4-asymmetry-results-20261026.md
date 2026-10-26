# NF4 asymmetry transfer — 26 October 2026

**The margin asymmetry appears on NF4, but the frozen top-1 transfer gate
fails.** The entity-row intervention moves He−H by **+19.7344**, compared with
only **+0.0469** for two−one. He nevertheless remains 1.5313 below H. No gain
search, edited continuation or PLE experiment follows this result.

## Scope and protocol history

This series asks whether the Q4K symbol/atomic-number asymmetry has an analogue
on the qualified local NF4 instrument before interpreting a PLE intervention.
It uses one previously observed symbol question and one atomic-number question,
the existing E4B NF4 setup and L21. It does not test a general editor.

Initial protocol `1b0b3407` is committed before any new forward. Its native
eligibility phase requires the digit 1 and a complete answer. Nine unpatched
forwards instead produce:

```text
Hydrogen's atomic number is one.<turn|>
```

The digit-only v1 eligibility is **false**, with a `verified_complete` receipt.
Those results stay unchanged. Before observing any intervention outcome,
amendment **`97302e66`** admits this exact native spelling, token ` one` (886),
with target ` two` (1156). Both are single tokens. No question, layer, gain rule,
donor or gate is changed, and native generation is not rerun. This is a
prospective intervention protocol amended after a format observation, not a
protocol fixed before every observation in the series.

The symbol boundary is taken from the previous verified NF4 native answer:

```text
Hydrogen's chemical symbol is $\text{H}$.<turn|>
```

Its prefix through the opening brace is replayed by original generated ids.
The number boundary likewise reuses the new freely generated native ids.
Native replays reproduce the saved logits bit for bit. These interventions
are **resumed-boundary diagnostics**, not integrated edited answers generated
from empty in one execution. No answer prefix is retyped for either primary
question. The symbol readout is bare `H`/`He` inside LaTeX; the number readout
is the word pair ` one`/` two`, unlike the earlier Q4K digit pair.

## Intervention and fixed amplitude rule

At the symbol boundary, compute the complete-subject helium minus hydrogen
post-block L21 contrast. The input differs only at the single entity token,
position 5. The assistant prefix remains the native hydrogen prefix.

The entity vector is multiplied by:

```text
norm(delta[5:prompt_end]) / norm(delta[5])
```

Computed once in NF4, this gain is **2.057271**, with float16-cast vector norm
**63.8967**. The Q4K gain 2.585 is not copied. Exactly the same vector is then
used on the number question and both off-target questions. No gain or layer is
searched. Symbol/number entity-row native and donor captures are bit-identical;
the predeclared float16 cross-shape tolerance is not needed.

Controls include zero interventions, full helium-subject donors, and a seeded
Gaussian direction with the same cast norm (0.1% maximum casting mismatch,
fixed before intervention). Both same-input zeros are bit-identical. Saved
before/after arrays verify the float16 addition and its single-row support.
Projected PLE inputs are captured at all 42 layers and remain unmodified.

## Results

Margins are target minus native token logits, read from the saved full vectors.

| Attribute | Native margin | Margin after R | Increment | Random increment | R top-1 | Full donor top-1 |
|---|---:|---:|---:|---:|---|---|
| Symbol, He−H | −21.2656 | −1.5313 | **+19.7344** | +1.0566 | **H** | He |
| Number, two−one | −18.0430 | −17.9961 | **+0.0469** | −0.0234 | **one** | two |

The full donors show that both targets are accessible at these exact boundaries.
The symbol increment dominates its random control and the number increment.
Here the native number margin is actually less negative than the symbol margin:
the pattern cannot be summarized as the number starting farther behind.
This does not identify the underlying mechanism.

The frozen passage criterion additionally requires **He to become top-1**,
the number to stay native, then complete released continuations. Only the
symbol top-1 condition fails among the core conditions. The runner therefore
stops after the 16 planned core forwards, without running the conditional
continuations. The margin asymmetry is a recorded outcome; it does **not**
turn the failed operational gate into a pass.

Lithium and carbon symbol controls retain Li and C as top-1 under native, R
and random conditions. Their answer prefixes are explicitly supplied by
substituting the element name into the saved symbol prefix. This tests local
value-boundary leakage on two examples, not free-chat preservation or a
selectivity estimate.

## Budget, verification and evidence

- Two sequential bounded services use **9 + 16 = 25 new forwards**, below the
  declared total maximum of 40 and the individual caps of 16 / 24.
- Both services exit zero. Runtimes are 24.0 s and 32.1 s. Final service logs
  report 7.6G / 7.5G memory peaks and no swap; 22G high, 24G maximum, 8G swap
  maximum and 30-minute limits remain intact.
- The synthetic prelaunch test covers hook support, exact residual and PLE
  application, zero controls, cleanup and cap refusal. It is not model evidence.
- The model-free checker returns `verified_complete` on both phases, including
  the **negative v1 eligibility** and **negative operational transfer gate**.
  It checks artifact hashes, saved argmaxes, source replays, ancestry,
  tokenizer decoding, exact additions, norm construction and memory limits.
- The same coding agent writes runner and checker: no independent verification.
- Sources: [`run_nf4_asymmetry.py`](../scripts/run_nf4_asymmetry.py) and
  [`check_nf4_asymmetry.py`](../scripts/check_nf4_asymmetry.py).
- Compact evidence: [`nf4-asymmetry-20261026.json`](research-data/nf4-asymmetry-20261026.json).
- Raw evidence is local under `nf4-asymmetry-20261026/{native,interventions}`.
  A ZIP with SHA-256 manifest includes raw captures, protocols, checkers and
  the previous symbol-source evidence. Its restoration is verified without
  model inference.

No weights, vindex or persistent patch change. No PLE swap is performed.
H1 and H2 remain untested. This negative gate is specific to the fixed NF4
procedure and these value boundaries; it does not show that NF4 cannot produce
He, nor invalidate the archived Q4K result. Different quantization, answer
format and token readouts prevent assigning the discrepancy to one cause.

## Next decision

The operational gate is deliberately respected: do not increase the gain to
force He. The intended mechanism question concerns unequal **margin effects**,
which are already observable here. A future PLE factorial test could therefore
use margin changes as its preregistered primary endpoint without requiring
an initial top-1 flip. That would be a new mandate/protocol, not a continuation
under the failed gate. No such test is launched in this series.
