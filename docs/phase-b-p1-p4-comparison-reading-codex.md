# P1-4 comparison reading, recorded before peer factors are available

P1-4 starts only after Claude's completion index and Codex's five complete
statuses exist. Before then, the only permitted peer file is
`solves-claude/completion.json`. Its bytes are sealed before Codex's first
solve. No peer source, factors, norms or traces are consulted beforehand.

The approved per-fact comparison is cosine(U) >= 0.95, cosine(V) >= 0.95,
and update Frobenius norms within 10%. Rank-one norm is
`abs(scale) * ||U||₂ * ||V||₂`; no dense update matrix is needed. Directed
cosines are reported as declared, without silently flipping factor signs
or renormalizing either candidate to improve agreement.

“Within 10%” has a denominator convention to disclose. Report the norm
ratio max/min, the relative difference against each side, and the symmetric
relative difference against their mean. If these conventions would change
the agreement decision at the threshold, retain it as an interpretation
difference for Pat rather than silently selecting a favourable convention.

Each algebra check uses the candidate's explicitly saved key, target and
**affine** base value Wk*+b. Verify factor dimensions/orientation, declared
scale, all recorded hashes and the unchanged model/covariance/template
inputs. Report absolute and normalized constraint error; Codex's declared
float32 tolerance is rtol/atol 2e-5. Preserve both clean-key and mean-key
witnesses when supplied; do not substitute one for the other's declared
constraint.

No factor or method is repaired to make the comparison pass. More than
half of five disagreements blocks P1-5 under A1; matching candidates remain
uncertified until Claude's separate full harness run. No model forward is
part of P1-4.
