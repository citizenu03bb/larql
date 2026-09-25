# Baseline Contract Audit v1 — result

The native chat contract changes the behavior of the two spent hydrogen prompts: LARQL and the local llama.cpp comparator produce exactly the same eight-token continuations, with the correct symbol `H` and atomic number `1` at token eight. Under raw completion, both runtimes can instead continue questionnaire formatting. This supports an input/output-contract confound in NBCP's immediate-next-token eligibility test; it does not establish the cause of all 15 failures.

NBCP-v1 remains closed at its original **1/16 eligibility result**. Its confirmation material was not accessed, no causal interventions were run here, and this diagnostic does not replace or qualify that result.

## Fixed outputs

Inputs are two post-result-selected, already spent questions from discovery: `The chemical symbol of hydrogen is:\n` and `The atomic number of hydrogen is:\n`. Each is tested raw and through the shipped native chat template. Literal `\n` below represents a newline. Every continuation is limited to eight sampled tokens, including any EOS token.

| Input | LARQL continuation | First token ID | llama.cpp continuation | First token ID |
|---|---|---:|---|---:|
| Raw symbol | `A) O\nB) H\n` | 236776 (`A`) | `**H**` | 1018 (`**`) |
| Raw atomic number | `1. 1\n2. ` | 236770 (`1`) | `A) 1\nB) ` | 236776 (`A`) |
| Native chat symbol | `The chemical symbol of hydrogen is **H` | 818 (`The`) | `The chemical symbol of hydrogen is **H` | 818 (`The`) |
| Native chat atomic number | `The atomic number of hydrogen is **1` | 818 (`The`) | `The atomic number of hydrogen is **1` | 818 (`The`) |

All LARQL responses stop at the eight-token limit. The reference raw-symbol response ends naturally with token 106 after four sampled tokens (`**`, `H`, `**`, EOS); its other three responses reach the eight-token limit. The chat outputs are truncated before the closing Markdown markers. Exact agreement refers to the entire saved token arrays, not just visible text.

The raw symbol response includes `H` among multiple choices; that is not proof it selects `H`. Conversely, the native-chat first token is `The`, not the factual answer, even when the correct value appears within the bounded continuation. A factual next-token gate therefore requires a separately validated answer-position contract.

## Execution and verification

- Frozen instrument commit: `de7917cce6e00964436cb837f42ab50759f973cd`.
- Manifest SHA-256: `a350c21b15b264c03d294845c23f3dfa347b612acd3ff20056c0ba8fb9243eee`.
- LARQL: 32 attempts and 32 completed forwards; all 262144-element logit vectors finite. Independent verification recomputed every SHA-256, greedy argmax, input prefix and stop condition from the raw f32 files.
- Both raw first-step logit vectors exactly reproduce NBCP's stored digests. All 17 vindex files matched the earlier execution manifest before execution.
- Reference: four requests, 28 sampled tokens, at most 32 prefill/decode calls. Total conservative call bound is **64**, below the frozen maximum of **68**. Unused budget expires.
- Reference tokenizer parity passed on all four complete input arrays. Requests passed integer tokens directly to `/completion`, with no chat re-rendering, warmup, cache, or speculative model.
- Python controller tests: 3 passed. Rust worker tests: 2 passed. Actual Rust metadata/tokenizer preflight passed with zero forwards. Independent evidence audit passed.
- Sequential bounded services exited successfully; actual model processes were confirmed absent afterward. LARQL: 12m35.613s wall time, 22G journal-reported memory peak and 2.5G swap peak. Reference: 29.801s wall time, 7.1G memory peak. These include startup and integrity checking and are not controlled throughput benchmarks.

Run evidence lives at `/home/pat/Projects/larql-research/baseline-contract-audit-v1/run-20260925`. Service invocation IDs are `675bf26d453940a7b39b45c881e8610d` (LARQL) and `3d8ce8a5c5a5493db618e126388b944c` (reference). The evidence package includes the manifest, source, complete LARQL logits, full reference responses, token parity, independent verification, and both service journals.

## Scope of the audit

The model-free audit finds byte-identical HF/vindex template, tokenizer and generation metadata. The 719 source language-model tensors are structurally accounted for by 718 manifest entries plus the separate embedding store, with no shape mismatch or extra manifest key. It found no concrete forward defect; structural checks do not establish numerical correctness.

The reference binary reports `ead3024`, but its source checkout has local adapter and cvec-dump modifications. The executable and seven local shared libraries are hashed, the tracked source diff and status are preserved, and the library hashes were checked again after execution. Clean-checkout equivalence is not claimed.

The GGUF Q4_K_M and vindex are different quantized artifacts: three sampled layer-0 planes have different bytes despite matching quantization types. The reference also uses cached autoregressive decoding while LARQL recomputes full prefixes. Raw-output divergence therefore cannot identify a LARQL bug. Agreement on two short chat prompts also cannot validate all numerical paths, long contexts, or other facts.

## Recommended next decision

Prepare a fresh, model-free causal protocol that uses the native chat contract and defines an answer-span readout and its token position explicitly. Establish eligibility on separately designated development material before freezing a new causal test and untouched confirmation material. Do not assume the factual answer is the first generated token, change the closed NBCP gate, or use these selected diagnostic examples as new holdout evidence.

A matched-weight, layerwise differential audit remains the appropriate way to resolve numerical fidelity if that becomes the research question. The current diagnostic supplies no basis for either declaring an inference bug or declaring the implementation fully correct. No further model run is started as part of this closeout.
