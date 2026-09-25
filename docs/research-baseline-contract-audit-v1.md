# Baseline Contract Audit v1

Status: frozen diagnostic design; no new causal claim. Authorization: user “go ahead” after the recommendation to audit the native input contract and compare short continuations on already spent prompts. NBCP-v1 remains closed at eligibility (1/16); no confirmation material is opened and no unused budget is transferred.

## Question and fixed material

Does the same unedited instruction-tuned Gemma 4 E4B produce different short continuations under raw completion and its shipped native chat format? Is its behavior consistent with a local llama.cpp comparator? This is a post-result diagnostic, not a held-out evaluation or a rescue of NBCP.

Exactly two spent discovery questions: the first canonical hydrogen symbol failure and the first canonical hydrogen atomic-number success. `cases.json` freezes their raw inputs and native user/model turns, in that order, with exact token IDs. The original prompts are unchanged; the native Jinja template trims the user text, including its trailing newline. Therefore this contrast bundles turn wrapping and template trimming; it does not isolate either mechanism. Thinking is disabled, no tools, and an assistant generation prefix is added.

## Budget and stop rule

One execution only, two sequential engines. Four cases per engine, greedy decoding with at most eight sampled tokens per case, stop on EOS. LARQL recomputes the full prefix for each token: at most **32 forwards**. Reference llama.cpp has at most four prefills and 32 decode calls: conservative upper bound **36 calls**, total **68**. Prompts fit within one prefill batch. No warmup, speculative model, retry, extra prompt, threshold tuning, or continuation past eight tokens. Any startup, parity, nonfinite, integrity, or historical replay failure terminates the affected run; a LARQL failure blocks the reference run. Unused calls expire. Startup health and tokenize requests perform no inference.

Before loading: clean committed source, metadata preflight, SHA-256 binding of source, executables, all vindex files, and GGUF. Every vindex hash must still match NBCP's original execution manifest. Before each forward/request, append and fsync an attempt event. Preserve complete LARQL f32 logits and reference responses. On the first step of each raw case, require the full LARQL logit digest to match NBCP exactly; this is a continuity check, not proof of numerical correctness.

Execute one model process at a time in separate user systemd services, `MemoryHigh=22G`, `MemoryMax=24G`, `MemorySwapMax=8G`, hard wall cap 5400 seconds. Inspect actual processes before each launch and after completion. Reference runs CPU only with explicit prompt token arrays, no re-rendering, no prompt cache, one slot, and no speculative decoder. Its tokenizer must reproduce all four frozen input arrays before the first completion.

## Interpretation fixed before inference

Report all eight truncated continuations verbatim, first tokens, and stop causes. This is descriptive: no statistical pass rate or correctness threshold is introduced. Eight tokens may end before an answer; such a continuation is incomplete, not an incorrect answer. A change with native chat shows sensitivity to this combined input contract, not general task competence. Similar outputs support only local behavioral agreement. Divergence cannot identify an engine bug because the local Q4_K_M GGUF and vindex have different quantized weight bytes; they are not matched-weight oracles. A later exact-weight, layerwise differential audit would be a separate study. No causal experiment follows automatically from this diagnostic.

## Static findings

Source HF checkpoint is `gemma-4-E4B-it`, an instruction-tuned model. “Unedited” does not mean pretrained base. HF and vindex tokenizer JSON, tokenizer config, chat template and generation config are byte-identical. Original NBCP used BOS plus raw text and did not apply native user/model turns. Both encoders produce one BOS.

Header-only source coverage finds 719 language-model tensors: 718 names/shapes accounted for by the three weight manifests, plus the dedicated `[262144,2560]` f16 embedding store. This is structural coverage, not numerical equivalence. Reviewed configuration includes 42 layers, width 2560, 8 query/2 KV heads, local/global head dimensions 256/512, sliding window 512, 18 shared-KV layers, partial global RoPE, GELU tanh, norm epsilon and final logit softcap 30. Static inspection has not established a concrete forward defect and is not a proof of implementation correctness.

Local llama.cpp source is `ead30241ed18643b3bea910b2ee203bbe955301e`. Comparator GGUF metadata identifies Gemma-4-E4B-It, Unsloth Q4_K_M, base repository google/gemma-4-E4B-it. Three inspected layer-0 planes (query, gate, down) have matching quantization types but different encoded bytes from the vindex. The binary and GGUF are bound separately in the execution manifest.
