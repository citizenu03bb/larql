# Where an entity edit lands decides what it changes

*Activation patching on one native fact in Gemma 4 E4B — follow-up to the LARQL multi-token and exact-surface posts. Status: closed exploratory series, 2026-10-26.*

## In short

- **Architecture sets a write deadline.** In Gemma 4 E4B, layers 24–41 reuse the keys and values computed at layers 22/23. A residual patch at layer 23's FFN or later changes the token being produced but cannot reach later positions; their logits stay bit-for-bit identical.
- **A one-token counterfactual is easy to flip and hard to own.** Adding a helium−hydrogen difference to the hydrogen token at layer 21 makes the model write "Hydrogen's chemical symbol is He." from an empty answer, on two phrasings. When the element's name comes *after* the symbol, the model tends to continue "…He is the symbol for helium": an entity swap, not a fact edit.
- **The same vector changes the symbol but not the atomic number.** At layer 21 it moves the He−H margin by +16.5 to +19.7 and the two−one margin by +0.05 to +0.8, while random directions of equal norm do almost nothing and a full helium subject flips both. Neither the form of the name (H1) nor the per-layer embedding path (PLE, H2) explains the gap. Injected at layer 8 instead, the vector moves the number by +9.6.
- **The selectivity does not survive a persistent form.** Applied statically to every hydrogen mention, the vector gives He, but the atomic number becomes "three". The number effect comes from the answer's own "Hydrogen" mention, which has already read the question: +15.6, against +0.05 on the prompt mention that precedes the relation.
- **Conclusion for persistent editing.** An activation delta keyed on the entity is not a selective, persistent fact edit in this form. Its apparent selectivity depended on the entity sitting *before* the relation. A persistent edit must act on the subject early and consistently across mentions, which is where weight-editing methods in the ROME/MEMIT family operate. That is the next step, with a certification harness built from the controls below.

## Setup

- **Model.** Gemma 4 E4B-it, in two instruments:
  - the LARQL native-Q4K vindex (Rust, CPU, full-prefix recompute, about 45 s per forward);
  - a local transformers + bitsandbytes NF4 bench on a GTX 1660 Ti (about 0.3 s per forward), qualified against LARQL on short cases.

  Numbers are not compared bit for bit across instruments.
- **Fact.** Hydrogen's chemical symbol, with a deliberate counterfactual H → He. He is helium's real symbol, which turns out to matter. Chat template, greedy decoding.
- **Intervention.** The difference between the residual for a question about helium and the same question about hydrogen, taken on the subject row at a chosen layer and added once (or statically, in the last series). No weights, vindex or persistent patch were changed.
- **Discipline.**
  - Every series froze its budget, stop rule, failure criterion, controls and reading before the first forward, in a committed protocol.
  - Each series has a model-free verifier, bit-exact replays of earlier captures and an archived SHA-256 manifest.
  - Most verifiers were written by the same agent as the runner. Two series (PLE, timing) were cross-checked by a different agent from raw artifacts.
- **Who did what.** The work was designed, coded, run and verified by AI coding agents (Codex and Claude Code) under the direction of the author. The author set mandates and budgets, decided at forks, and convened audits and model panels.

## 1. The preregistered causal-patching test came back negative

On the edited multi-token artifact of the earlier posts, replacing the pre-FFN residual at the last position (240 forwards) did not nominate layer 11 or layer 13 as a mediator of the entity–relation dependence. The late-layer effects were large but not specific. The test is mechanically valid: replays identical, self-patch drift 0.0.

## 2. The K/V-sharing deadline

On the LARQL instrument, a vector re-applied after layer 31 at an earlier position left all 262 144 logits of each following token bit-for-bit identical. Residual plus K/V at layer 22 reproduce the effect exactly; neither alone does. On the NF4 bench, Gemma 4 E2B (35 layers, layers 15–34 reuse the K/V of 13/14) shows the same boundary: a perturbation after layer 14 leaves the next position's logits and shared K/V bit-identical.

**Consequence.** Any edit meant to influence later tokens must be written at layer 22 or earlier on E4B. This also explains an earlier negative of the slot-based line, whose injections sat at layers 25–32.

## 3. H → He, and the entity-substitution trap

On LARQL Q4K, adding the contrast once at layer 21 at the native proposal of "H", then releasing, gives "Hydrogen's chemical symbol is He." with end of turn, generated from an empty answer, on two phrasings:
- on one, on the hydrogen row alone;
- on the other, on the prompt span from hydrogen onward.

Both prompts ask for the element's name before the symbol. When the name comes after, the emitted "He" drags the continuation to "…symbol for helium". An ablation shows this drift is carried by the emitted token itself, not by the vector.

A norm-matched 2×2 on the resistant phrasing (verified) shows that both location and amplitude matter:

| Rows patched at L21 | Gain | Proposal | He − H change |
|---|---:|---|---:|
| entity only | 1 | LaTeX opener, then H | +13.1 |
| entity only, amplified to the span norm | 2.585 | **He** | +16.5 |
| prompt span | 1 | **He** | +20.0 |
| prompt span, attenuated to the entity norm | 0.387 | LaTeX opener, then H | +5.1 |

(Native He − H at that boundary: −12.42.)

## 4. Symbol yes, atomic number no

**Same vector, same layer** (Q4K, gain 2.585):

| Question | Native margin | Change from the delta | Change from a random direction of equal norm | Full helium subject |
|---|---:|---:|---:|---|
| symbol (He − H) | −12.42 | **+16.5** → He | +2.1 | He |
| atomic number (2 − 1) | −20.34 | **+0.8** → still 1 | −0.1 / −0.1 | 2 |
| atomic mass (4 − 1) | −26.87 | +5.6 → still 1 (4 rises from 48th to 4th) | — | 4 |

On the NF4 bench the same picture holds: +19.7 on the symbol, +0.05 on the number.

## 5. What does not explain it, and what partly does

- **H1, form of the name** ("He" is how "helium" starts). Eight donors were tested: four whose symbol starts their English name (He, Li, Ca, Ni) and four that do not (Na, K, Fe, Cu). The preregistered reading is **undetermined**: with the answer already saying "Hydrogen's chemical symbol is", only the helium full donor flips its own symbol, so the normalisation has one eligible donor. A post-hoc matrix (not preregistered) shows each donor's delta raising its own symbol more than any other in 8 of 8 rows, Latin-derived symbols included. This weakens a spelling-only account.
- **H2, the per-layer embedding (PLE) path.** Replacing hydrogen's PLE inputs with helium's at all 42 layers moves the symbol by +9.7 but the number by only +0.14. Combined with the residual delta, the effects are not additive (symbol interaction −11.0). H2 is not supported for the number.
- **H3, timing.** At a common absolute norm, the delta moves the number by **+9.6 at layer 8**, −0.004 at layer 14 and +0.05 at layer 21; the symbol by +14.6, +2.4 and +19.7. Random controls are at most +1.0. The preregistered reading is *partial/local*: layer 8 passes, the ordered pattern fails. Relative doses differ by layer (0.96, 0.47 and 0.62 of the row norm), and an early pulse changes the whole later history, so this does not isolate a K/V route.

## 6. A persistent form loses the selectivity

**Phase 3, static patch.** The layer-21 vector was added to every hydrogen token at every forward, prompt and answer alike, with the gain calibrated on two observed boundaries by a fixed rule.

| Gain | Symbol | Atomic number |
|---:|---|---|
| native | H | one |
| 1.0 | He | **three** |
| 1.5 | He | three |
| 2.0 | He | one |
| 3.0 | He | one |

The selectivity pre-check fails at the selected gain, so the free-generation tests were not run, by rule.

**Phase 3b, row split.** The same vector at layer 21, applied to one row at a time:

| Row patched | Symbol change | Number change |
|---|---:|---:|
| prompt "hydrogen", before the question is read | +19.7 (H) | **+0.05** (one) |
| answer "Hydrogen", after the question is read | +45.0 (He) | **+15.6** (three) |
| both, half gain | +41.8 (He) | +16.5 (three) |

A single application already gives "three", not "two", so a simple helium-ward step is not supported. Why "three" appears is unexplained.

**Reading.** The selectivity seen at layer 21 was a property of *where the entity sits*. A mention that has already read the relation carries the vector to both read-outs. An edit keyed on the entity fires on every mention, and natural questions ("What is the atomic number of hydrogen?") put the entity after the relation.

## What is not claimed

- No knowledge edit, no persistent edit and no generalisation across facts. One fact, a handful of phrasings and one trial per case.
- No entity–relation mechanism established, and no unique K/V mediation.
- The verifiers are mostly written by the agent that ran each series; this is not independent replication.
- The resumed-boundary diagnostics are not from-empty generations unless stated.

## Next

A method-agnostic **edit certification harness** that turns the controls above into a `CERTIFIED / REFUSED (reasons)` verdict:
- unseen phrasings in free generation;
- other attributes of the same entity;
- mentions before and after the relation;
- random directions of equal norm;
- full-donor ceilings;
- off-target questions;
- persistence after reload.

Then a weight-editing method at the subject token, early in the network, evaluated by that harness on a small set of facts. First counterfactuals, then real model errors.

## Sources

- **Branch `research/exploratory-natural-fact-edit`:**
  - Q4K series, journal (French) and status;
  - NF4 bench, asymmetry, PLE factorial and H3 timing reports (`docs/nf4-*-results-20261026.md`).
- **Branch `research/nf4-h1-name-form`:**
  - H1 report and its post-hoc matrix;
  - Phase 3 and 3b reports;
  - cross-checks of the PLE and H3 runs (`docs/research-data/*crosscheck-claude.json`).
- Every series has its frozen runner, model-free checker and compact evidence JSON. Raw traces and SHA-256 archives are kept locally.
