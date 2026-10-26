# NF4 Phase 3 static L21 patch — 2026-10-26

**Outcome: stopped at calibration by the preregistered rule.** Applied statically to every hydrogen token, the L21 entity delta flips hydrogen's symbol to He at every gain tried. At the smallest such gain, it also changes hydrogen's atomic number: the top-1 becomes " three", not " one". The frozen selectivity pre-check fails, so no test service ran.

## Protocol

The protocol was frozen in `scripts/run_nf4_phase3_static.py` (commit `0406b231`) before any forward:
- **Vector:** the archived L21 residual (`nf4-asymmetry-20261026`, float16 norm 63.897), times a gain g.
- **Static rule:** at every forward, the vector is added to the post-L21 output of every hydrogen token (`13076, 50604, 70599, 154624`), prompt and answer alike. There is no donor pass and no release. PLE stays native.
- **Calibration:** on the two already observed resumed boundaries (symbol and number), with g in {1, 1.25, 1.5, 2, 2.5, 3}.
- **Gain rule:** g\* is the smallest g whose symbol top-1 is He. Testing proceeds only if the number top-1 stays " one" at g\*.

Both boundaries contain two hydrogen tokens, the prompt's " hydrogen" (row 5) and the answer's "Hydrogen" (row 30), so the static rule patches both rows.

## Verification

- **Model-free checker:** `scripts/check_nf4_phase3_static.py` returns `verified_complete`.
- **Checks:** artifact hashes; patched rows equal to the hydrogen-token positions; exact requested vectors; recorded increments; native replays bit-identical to the archived logits; calibration rule recomputed.
- **Authorship:** the checker was written by the same agent as the runner (Claude Code), so this is not an independent replication.
- **Resources:** 14 forwards in one service, 31.6 s, 7.8G peak, no swap; the shared heavy-model lock was held.

## Results

Margins are target minus native logits: He − H for the symbol, " two" − " one" for the number.

| g | Symbol top-1 | He − H | Number top-1 | two − one |
|---:|---|---:|---|---:|
| native | H | −21.27 | one | −18.04 |
| 1.0 | **He** | +25.93 | **three** | −0.50 |
| 1.25 | He | +25.72 | three | −1.25 |
| 1.5 | He | +25.29 | three | −2.17 |
| 2.0 | He | +24.77 | one | −8.20 |
| 2.5 | He | +21.84 | one | −14.40 |
| 3.0 | He | +12.61 | one | −16.56 |

Other tokens near the top:
- **Symbol at g = 1:** "He" 25.0, then "he" 9.1 and "Hel" 8.2.
- **Number at g = 1:** " three" 22.1, " " 18.6, " one" below the top four.

## Reading

**Preregistered reading.** g\* = 1.0, and the number top-1 at g\* is " three", not " one". The frozen rule says STOP. The selective static edit was not tested and is **not supported** in this form.

**What the stop shows.** Patching the answer's own "Hydrogen" row as well as the prompt row breaks the selectivity seen when only row 5 was patched at one boundary. In that earlier condition, the number moved by +0.05. Here, the " two" − " one" margin moves from −18.0 to −0.5 at g = 1. The answer-row mention has already read the relation in its context; adding the identity delta there reaches the number read-out.

**Post hoc, one case, speculative:** "three" is lithium's atomic number. One possible reading is that the helium−hydrogen difference acts like a step along an element order, applied here on two rows. The symbol does not move to "Li", and the effect disappears above g = 2. This is an observation, not a claim.

## Limits

- One NF4 model, two resumed boundaries and one trial per gain.
- The static rule fires on both mentions. A prompt-only rule would be a different patch, and it is not what a persistent edit keyed on the entity would do.
- No free-generation tests ran.
- No weights, vindex or persistent patch were written.
