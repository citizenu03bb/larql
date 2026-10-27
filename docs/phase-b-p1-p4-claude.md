# Phase B P1-4: Claude's cross-check and the cause of the disagreement

*Claude Code, 2026-10-27. This page cross-checks Codex's P1-4 (`docs/phase-b-p1-solves-and-p4-results-codex.md` on `research/phase-b-rome`, `ac0ef8ec`). It runs no model. P1-4 has started, so I have read Codex's solve outputs and metadata, but not its port.*

**Mandate line.** A positive control is only useful if its failure is understood. This page shows, model-free, why the two solves disagree, and on which side the error lies.

## The gate: 0 of 5, confirmed

- **I recomputed the gate on my own** with `tools/edit_cert/scripts/p1_p4_xcheck_claude.py`, from both agents' arrays and C17. The output is `docs/research-data/phase-b-p1-p4-xcheck-claude.json`.
- **Every number in Codex's table is reproduced:**
  - cos U is 0.59 to 0.75;
  - cos V is 1.0000;
  - the norm ratio is 1.64 to 424.
- **The context keys are bitwise identical.** Codex's constraint key is exactly the key of the clean template `{}`.
- **The A1 reading is "disagreeing solves".** P1-5 does not run.

## The cause, split exactly into two factors

Codex's U is parallel to its additive shift δ: the cosine is 1.0000 on all five facts. Each norm ratio therefore factors exactly into two parts:
- the **objective factor**, ‖δ_Codex‖ / ‖v\*−z0‖_Claude, which is how far each optimisation moved the output;
- the **key factor**, (u·k_mean)/(u·k_clean), which comes from the constraint key. Here u is the unit C⁻¹k_mean direction that both V share.

| Fact | Subject's last token in the clean prompt | Objective factor | Key factor | Ratio |
|---|---|---:|---:|---:|
| BMW | position 1 ("BM", "W") | 4.10 | 1.73 | 7.08 |
| Germany | position 4 | 1.66 | 0.99 | 1.64 |
| Gretzky | position 4 | 3.77 | 0.97 | 3.67 |
| Nintendo | **position 0** | 6.94 | **61.0** | 424 |
| Shakespeare | position 1 | 2.18 | 1.10 | 2.39 |

- **The objective sets U's direction (cos 0.6–0.75) and a factor of 1.7 to 6.9 on the norm.** It affects all five facts.
- **The key matters mainly for Nintendo.** Its subject is the clean prompt's first token, GPT-2's first position. There u·k_clean = 0.113, against 6.9 for the mean key, which inflates Codex's U by a factor of 61.

## Which side follows the published ROME

I read the published code (`kmeng01/rome`, `rome/compute_v.py`, and `hparams/ROME/gpt2-xl.json`) as text. I did not run it.
- **The shift is added.** The hook adds δ to each prompt's MLP output at the subject's last token.
- **The weight decay** is `v_weight_decay · ‖δ‖ / ‖target_init‖²`. Here `target_init` is the output on the first prompt, the clean one.
- **The clamp** is 4·‖target_init‖.
- **The constraint key** is that of the request's own prompt, with no context: `right_vector = (target − cur_output) / (cur_input · left_vector)`.

**Codex's solve follows the code on all four points. Mine departs from it on three:**
1. **I replace the output with z instead of adding δ.** This follows the paper's notation `m := z`. Under replacement every context gets the same output; under addition each context keeps its own native output plus δ.
2. **I penalise ‖z−z0‖²/‖z0‖², which is squared, with z0 taken from the mean key.** The code penalises ‖δ‖/‖target_init‖², unsquared, with target_init taken from the clean prompt. The squared form was my own choice, recorded in the solve's `decisions`. My penalty is much stronger, which explains why my shifts are smaller.
3. **My constraint uses the mean key k\*,** as in the paper's equation. The code uses the clean prompt's key.

**A1 was under-specified.** It fixed "the published hyper-parameters", but not which formulation, the paper's or the code's. The paper's numbers come from the code. The gate did its job: it caught a divergence in the specification. Neither agent made an algebra error, since both pass their own constraint.

## What this changes

- **The B2 negative stays suspended,** under the protocol's "disagreeing solves" reading. The divergence is in my solve, not in Codex's solver.
- **The ceiling of P1 is lower than A1's top reading.**
  - The answer-only witness W1 is excluded at qualification for BMW and Gretzky, as Codex points out. I checked this in `phase-b-p1-v2-qualification-index-claude.json`.
  - At most 3 of the 5 facts can therefore be fully certified. The best reachable reading is **"partial"**.
- **The ten candidates and the failed gate are kept as they are.** No candidate is chosen, rescaled or certified.

## Recommendation: amendment A3, to approve before any forward

- **The reference specification is the published ROME code** (`compute_v.py` and `compute_u.py`, at a pinned commit), not the paper. This covers the additive δ, the unsquared decay over ‖target_init‖², the clamp at 4·‖target_init‖ and the clean-prompt constraint key.
- **Claude rewrites its solve to that reference** from the published code, without reading Codex's port. The two solves remain two independent implementations, now of one fixed reference. The independence of the specification, by contrast, ends here, by design.
- **Then 5 new Claude solves** (2 310 sequence-forwards, same caps), then P1-4 again with unchanged thresholds. Codex's existing solves are the comparison; Codex does not re-solve.
- **The budget goes over the design's 10 000.** This page first estimated about 11 600. The exact count, with P1-5 measured from the bench's plan, is in `docs/phase-b-p1-amendment-a3.md` §6: at most 13 364 since P1 v2. **This requires Pat's approval.**
- **Why it is worth it.** Even capped at "partial", P1-5 would give the first certified real edit of the project, on a method the literature reproduces. It is also the only way to lift B2's suspension.
