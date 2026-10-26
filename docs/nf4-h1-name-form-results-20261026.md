# NF4 H1 "form of the name" test — 2026-10-26

**Outcome: undetermined under the preregistered reading.** The full-donor ceiling selected the donor's own symbol for helium only. For the seven other donors, the full donor still answered `H`. With one eligible donor, the frozen reading cannot be applied. No rerun with a changed design was made.

## Question

On the verified NF4 Gemma 4 E4B symbol boundary, an entity-row delta at the post-L21 residual moves He−H by about +19.7 but barely moves hydrogen's atomic number. H1 asks whether the symbol read-out leans on the *form* of the donor's name ("He" is how "helium" starts). This test covers H1 only; the PLE hypothesis (H2) is a separate line run by another agent.

## Protocol

Frozen before any forward in `scripts/run_nf4_h1_name_form.py` (commit `e6aab9f4`):
- **Boundary and hook.** Same boundary and hook as `scripts/run_nf4_asymmetry.py`: the prompt "Give hydrogen's chemical symbol in one short sentence, with the element's name before the symbol." plus the saved native assistant prefix "Hydrogen's chemical symbol is ".
- **Donors.** Eight donors, each a single-token name aligned at position 5 with a single-token symbol:
  - four whose symbol starts their English name: helium He, lithium Li, calcium Ca, nickel Ni;
  - four whose symbol does not: sodium Na, potassium K, iron Fe, copper Cu.
- **Delta.** For each donor, a full-donor forward, then the donor-minus-native row-5 delta rescaled to the norm of the verified helium delta (63.897). Helium reuses the verified delta exactly.
- **Shared controls.** Native replay, zero, and two random directions of the same norm.
- **Metric.** ρ = (residual margin − native margin) / (ceiling margin − native margin), with margin = logit(symbol) − logit(H).
- **Eligibility.** A donor is eligible when its ceiling selects its own symbol. At least three eligible donors per group are required.

## Verification

- **Model-free checker.** `scripts/check_nf4_h1_name_form.py` returns `verified_complete`. It checks:
  - artifact hashes and ordinals;
  - that donor prompts differ only at position 5;
  - the exact fp16 application of every delta, all other rows unchanged;
  - the recomputed margins and reading;
  - the second random direction, reproduced from its seed.
- **Replays of the verified asymmetry run, all bit-identical:** native, zero, helium donor, helium delta and the first random direction.
- **Authorship.** The checker was written by the same agent as the runner (Claude Code); this is not an independent replication.
- **Resources.** 20 forwards, 41 s of service time, 8.3G peak memory, no swap; the shared heavy-model lock was held.

## Results

| Donor | Group | Native margin | Ceiling margin | Ceiling top-1 | Delta increment | Max random increment | ρ (descriptive) |
|---|---|---:|---:|---|---:|---:|---:|
| helium | prefix | −21.27 | **+1.97** | **He** | +19.73 | +1.38 | 0.85 |
| lithium | prefix | −27.42 | −0.17 | H | +13.84 | +1.23 | 0.51 |
| calcium | prefix | −28.27 | −1.47 | H | +5.20 | −0.04 | 0.19 |
| nickel | prefix | −36.70 | −1.33 | H | +4.62 | +0.70 | 0.13 |
| sodium | non-prefix | −29.75 | −2.61 | H | +4.37 | +0.96 | 0.16 |
| potassium | non-prefix | −30.22 | −4.92 | H | +9.03 | +0.24 | 0.36 |
| iron | non-prefix | −31.38 | −3.77 | H | +2.39 | +0.60 | 0.09 |
| copper | non-prefix | −32.18 | −2.28 | H | +9.42 | +0.23 | 0.32 |

The delta never changed the top-1 token; every residual forward still selected H.

## Reading

**Preregistered reading.** Only one donor, helium, is eligible, so the outcome is **undetermined**. This is a design limit: the answer prefix already names "Hydrogen", and that pull is strong enough that a full donor subject only overrides it for helium.

**Descriptive observations,** not a test of H1:
- Every delta moves its own margin well beyond the random directions of equal norm.
- The ρ values overlap between groups: potassium (0.36) and copper (0.32) exceed calcium (0.19) and nickel (0.13).
- The prefix-group mean is 0.42 against 0.23 for the non-prefix group, carried by helium and lithium.
- That helium is the only full donor able to override "Hydrogen's chemical symbol is" may itself be informative, since "He" shares its first letter with "H" and "Hydrogen". This is not tested.

No H1 conclusion is drawn.

## Limits

- One question, one template and one resumed boundary; one trial per case.
- Four donors per group.
- The ceiling design kept the hydrogen answer prefix; a donor-named prefix would answer a different question.
- Confounds left open: symbol-token frequency, chemical family, name length.
- No gain search, no new layer, no PLE change.
- No weights, vindex or persistent patch written.
