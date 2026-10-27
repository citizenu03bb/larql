# Global synthesis — Phase B next step, three blind runs, 10 answers from 9 models

Claude Code, synthesizer, non-contributor. Draft, audit me.
Runs: 20261027-051554 (gemini-3.8-flash, mistral-large-4 s1, deepseek-flash), 20261027-054948 (mistral-large-4 s2 served by Le Chonk, glm), 20261027-052427 (gemini-4-argon, kimi, qwen, grok, minimax). Same question, unchanged. Codex and Opus excluded as authors.
Weighting: mistral s2 weighted up (frontier reasoning, per Pat). No vote: counts below map positions, they do not decide.

## Framing defects (found after launch, by the synthesizer)
- C3 was described as the answer-only mention only. In the harness it is "mentions before and after the relation": it fails whenever the new value is missing in either subject order, so it echoes C1 on the 11 non-flipping edits, and the answer-only substitution (Verdi, Volkswagen) is one of its failure modes. This weakens minimax's "relax C1" rescoring (C3 would still fail) and several readings of "C3 FAIL" as substitution everywhere.
- The model-free algebra cross-check (Claude, 15/15: k*, U0, V0, W′k* = v* within 2e-5, hashes, C against B1) was not mentioned. kimi's proposed cheapest control is therefore already done; mistral s2's "verify ΔW in the served weights" is only partly covered (the algebra is checked on saved tensors, not inside the served forward).

## 1. Convergences
- 10/10: close B2 on the stop rule; keep the 20 test facts sealed; no MEMIT/AlphaEdit now; claims bounded to Gemma 4 E2B-it NF4, ROME rank 1 on down_proj, blocks 4/8/12, these hyper-parameters, this C, 5 dev facts.
- 10/10: the showcase is the bench, the pre-registration and the stop rule kept, not a green edit (grok: "Forcer un succès avant un entretien casserait exactement ce qui est montrable").
- 9/10: the two-agent check is internal validation, not independent (kimi, qwen, grok, glm, mistral ×2, deepseek, gemini-4-argon, minimax).
- 9/10: the off-domain covariance is the prime suspect; NF4, 4 short prefixes and the 4× clamp are the next ones. All label them speculative.
- 8/10: the cheapest discriminating measurement is the key geometry of subject vs neighbours (cos k*, or the update's activation on neighbour keys), without generation (gemini-flash, deepseek, mistral s2, glm, gemini-4-argon, qwen, grok, kimi as part of its control).

## 2. Divergences
- **Generative step before the write-up?** None beyond capture-only: gemini-flash, qwen, mistral s2 (plus logit/rank probe). One bounded re-edit with a better C: grok (Beethoven L8 only, in-domain C), gemini-4-argon (4 C1-pass cases, Wikipedia C, 30 prefixes; needs a download that Q3 did not approve), deepseek (4 edits × {C = I, 64 prefixes}), mistral s1 (C = I on 15). B2b: kimi (preferred, 15 edits max, failure if no block reaches 2/5 CERTIFIED), minimax (conditional, L8 only).
- **B2b itself.** For: kimi, minimax (conditionally). Against, each for its own reason: mistral s2 (targets specificity, the minority failure; efficacy fails on 11/15), glm (tuned against a failure already seen on the same 5 facts), gemini-4-argon (treats the C6 symptom; a large v* keeps C3 and C2), qwen (would look like cherry-picking after 0/5), grok and deepseek (cause not diagnosed).
- **Is the bench or the implementation on trial?** A positive control is a prerequisite for interpreting the negative: grok ("aucune calibration de l'implémentation sur un modèle où ROME réussit"), glm and minimax (canonical GPT-2 XL Eiffel → Rome replay), mistral s1/s2, qwen (nonce sentinel edit, manual audit of a sample). Phenomenon judged real already: glm (matches the pre-registered reading), grok and minimax (published ROME failure signature; C4 inert rules out a norm effect).
- **Reading C2 NOT_EVALUATED.** A masked real degradation (hedging, EOS collapse): gemini-flash, gemini-4-argon. Not evaluated means neither preserved nor destroyed, and must be reported so: kimi, qwen, grok, glm, deepseek.

## 3. Draft (audit me)
- [certain] Close B2. Report the gate as a per-control vector, not "0/5" alone: at L8, two C1 PASS with C2 NOT_EVALUATED; every C1 PASS edit fails C6 (deepseek, kimi, qwen, grok); C3 fails by entity substitution on two of them (Beethoven L12, Toyota L8), by attributes on Bolt L12, and is NOT_EVALUATED on Beethoven L8 (corrected 2026-10-27 after Codex's remark).
- [certain, checked] Efficacy is the majority failure (11/15 C1 not PASS); specificity fails on all 4 that flip. B2b addresses only the second (mistral s2), on the same facts (glm). It should not be the next step.
- [plausible, 8/10] The next measurement is capture-only: neighbour-key activation of the stored update (with C and C = I), plus the target-token rank probe (mistral s2), plus a served-forward check that down_proj(k*) = v*. Tens to a few hundred forwards, no solve, no Codex quota.
- [plausible, 6/10] The strongest remaining limitation is the absence of a positive control: the implementation has never been run where ROME is known to work. A canonical replay (GPT-2 XL, CounterFact) is the external answer (glm, minimax, grok); its answer format must be designed, since the harness reads chat answers [inference synthétiseur].
- [inference synthétiseur] Factual errors to discount: minimax says ROME was documented on 7B+ (it was GPT-2 XL 1.5B and GPT-J 6B) and calls the failure "uniform" (4 edits flip); grok says shared K/V removes most of MEMIT's layer range (blocks 0–13 remain); gemini-4-argon's "15 min GPU" with a Wikipedia C ignores the unapproved download.
- [plausible] Reframes worth keeping for the write-up: REFUSED does not mean "not edited" (kimi, glm); "persistent and effective, not safe" on 4 edits (glm); a single rank-1 site may be the wrong unit of edit (glm); an explicitly routed adapter may be more governable for agent memory than weight surgery (gemini-4-argon); the dev facts are famous and densely related, which makes C6 hard (qwen); E2B's allowed site is not a causal site (grok, mistral s2).

## 4. Verbatim
raw/*.txt in each run directory.
- gemini-4-argon: "Projeter u hors de quelques voisins (B2b) ne traite que le symptôme local (C6)."
- kimi: "La principale est d'assimiler REFUSED à « le fait n'a pas été édité »."
- qwen: "Relancer B2b/MEMIT après 0/5 sans diagnostic ressemblerait à du cherry-picking."
- grok: "Sans contrôle positif de l'outil, un négatif sur un modèle nouveau est ininterprétable."
- minimax: "assouplir C1 à une formulation dans un seul ordre, et voir si l'un des 15 edits certifie sous cette règle relâchée."
