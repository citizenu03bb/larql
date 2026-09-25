# Natural factual edit — exploratory journal

2026-09-25. The user explicitly switches this work to adaptive exploration. Prompts, positions, gains and candidate interventions may change in response to results. Earlier confirmatory studies remain closed. This session has one model load, at most 128 full-prefix forwards and 60 minutes of runtime, under MemoryHigh=22G, MemoryMax=24G and MemorySwapMax=8G. All trials are exploratory; no untouched evaluation is claimed.

Goal: make a natural answer say the counterfactual hydrogen symbol “He” instead of “H”, then inspect neighboring questions. Start with a donor-minus-recipient residual direction at L31, try L41 if needed. Apply at an explicitly chosen answer position and preserve that intervention in subsequent full-prefix recomputation. Generation uses plain greedy argmax: no token mask, target-token override, or postprocessing. This is a temporary activation intervention, not yet a persistent weight edit or an automatically learned semantic detector.

The small queue worker keeps the model loaded while individual trial jobs are added. Every forward and observed output goes into one session ledger; this journal records only decisions and findings.
