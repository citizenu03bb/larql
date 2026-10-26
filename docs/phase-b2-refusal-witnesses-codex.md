# Independent sufficient witnesses for refusal

The frozen checker at `1d6af9e8` recomputes C1/C2, not every control.
A separate post-comparison extension, `xcheck_b2_refusal_witnesses.py`,
checks each complete reported verdict by the sufficient condition in
the written verdict rule: one mandatory control that cannot PASS makes
the whole verdict REFUSED. It uses no harness judging imports or code.

The first output supports all 13 then-complete REFUSED verdicts. An agreed
independent C1 or C2 non-PASS is the witness when available. For Beethoven
L8/L12, C1 passes and C2 is disputed: a missing native-qualified attribute
order means C3 cannot PASS under its explicit both-order rule. This proves
refusal without relying on Pat's later C2 amendment or reconstructing all
C3 sub-outcomes. The output labels this witness NOT_PASS, not a fabricated
exact C3 status.

This extension was written after those peer verdicts were visible; it is
not presented as another blind freeze. The underlying C1/C2 reading code
and its pre-comparison output remain unchanged. Three model-free logical
edge-case tests pass. The extension cannot validate a positive CERTIFIED
case, does not claim exhaustive C0--C8 status replay, and reports any
unsupported or differing verdict instead of manufacturing a witness.
