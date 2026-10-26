# Local NF4 bench qualification — 26 October 2026

The existing local transformers/bitsandbytes setup runs short Gemma 4 E2B and
E4B inference without training. Both models complete the native hydrogen answer.
The E2B intervention test agrees with its K/V-sharing boundary on this example.
This qualifies a local instrument; it does not reproduce the H→He effect or
establish a knowledge edit.

## Frozen protocol and execution

The runner's protocol is committed before loading either model, at `2323e1c3`.
Two sequential user services enforce 22 GiB memory high, 24 GiB memory maximum,
8 GiB swap maximum and 30 minutes each. The runner separately refuses calls
after 24 forwards or its 29-minute deadline. Planned maxima are 21 E2B and 18
E4B forwards. Actual counts are **21 and 14**, 35 total. Both services exit zero.

The question is:

> Give hydrogen's chemical symbol in one short sentence, with the element's
> name before the symbol.

The local chat template, with thinking disabled, produces 30 prompt tokens and
an empty assistant answer. Both models generate the same twelve-token answer,
including the end-of-turn token:

```text
Hydrogen's chemical symbol is $\text{H}$.<turn|>
```

No answer tokens are supplied or forced. Generation is greedy and unpatched.
E2B reaches EOS on the twelfth and final allowed token; E4B has a sixteen-token
cap. Replaying the prompt for timing reproduces its logits bit for bit.

## Timing and resources

One full-prefix warmup precedes two timed full-prefix forwards. Timings include
the CPU PLE lookup, GPU forward and last-position vocabulary projection, with
CUDA synchronization. They exclude model loading and writing the evidence.
`use_cache=False`: these are **not cached decoding times**.

| Model | Warmup | Timed forwards | Mean | Peak GPU allocation |
|---|---:|---:|---:|---:|
| E2B NF4 | 0.512 s | 0.174 / 0.173 s | **0.174 s** | 1.69 GiB |
| E4B NF4 | 0.837 s | 0.321 / 0.315 s | **0.318 s** | 3.22 GiB |

Loading takes 17.1 s and 26.4 s respectively; complete service runtimes are
37.7 s and 50.1 s. Final service logs report memory peaks of 15.9G / 21.9G and
swap peaks of 0 / 437.8M. Cgroup accounting includes CPU-resident modules and
charged file cache; GPU allocation is a separate quantity. Limits stay intact.

Hardware: GTX 1660 Ti, with NF4 double quantization and float16 compute. PLE
embeddings and unused multimodal towers remain on CPU, following the existing
feasibility setup. Attention is eager, with deterministic settings, eval mode
and no gradients. Versions: torch 2.14.1, transformers 5.19.0, bitsandbytes
0.50.2, accelerate 1.15.0, numpy 2.5.3.

These are two timing samples on one short prompt. They do not estimate long
sequence performance or an end-to-end speedup over LARQL. NF4 is a different
numerical instrument from Q4K; no bitwise cross-engine claim is made.

## E2B boundary diagnostic

The installed implementation and the runtime module flags identify sliding
K/V source L13 and full-attention source L14. L15–L34 reuse these states.
All layer numbers are zero based.

Every condition uses the same prompt plus its first freely generated token,
`Hydrogen`. A fixed seeded Gaussian vector is added at the **last prompt row**,
after one decoder block, including its FFN and PLE branch. Its norm is fixed
at one quarter of the native L14 row norm: 17.9141. The same vector and norm
are used at L12, L13 and L14. There is no amplitude search.

This is a teacher-forced two-position diagnostic. Its second input token is
held identical even when the intervention changes the prediction at the first
position. It is not a free edited continuation or an entity-row fact edit.

| Intervention | Max current-logit change | Max next-position change | Shared K/V |
|---|---:|---:|---|
| Post-L12 | 20.0508 | 4.4746 | Sliding and full change |
| Post-L13 | 10.5391 | 1.1094 | Sliding identical, full changes |
| Post-L14 | 9.8574 | **0, bit identical** | **Both bit identical** |

The native repeat and zero-gain controls at L12/L14 reproduce both saved logit
rows and all shared K/V exactly. Every hook fires once, modifies only the
chosen row, and its float16 addition is checked from saved before/after arrays.
The pre-boundary positive controls change the later position; the late patch
is active at its own position while leaving the later position untouched.
The preregistered boundary prediction passes on this case.

## Evidence and limitations

- Runner: [`run_nf4_local_bench.py`](../scripts/run_nf4_local_bench.py).
- Model-free verifier: [`check_nf4_local_bench.py`](../scripts/check_nf4_local_bench.py).
- Compact evidence: [`nf4-local-bench-20261026.json`](research-data/nf4-local-bench-20261026.json).
- Both real runs return `verified_complete`. Checks include full saved
  logits/argmax, ancestry from an empty answer, EOS, tokenizer decoding,
  artifact hashes, exact application, replays and cgroup bounds.
- The same coding agent writes runner and verifier: **not independent
  verification**. The verifier later adds tokenizer decoding and memory-limit
  checks; the intervention protocol and outcome criteria remain unchanged.
- A separate synthetic CPU dry-run exercises orchestration, hooks, deadline,
  budget refusal, EOS and truncation. It is labeled synthetic and is not model
  evidence. The older EXP stub targets LARQL's subprocess protocol and cannot
  test torch hooks directly.
- Local raw evidence lives in `nf4-local-bench-20261026/{e2b,e4b}` under the
  research evidence root, with summaries, sessions, arrays and service logs.
  A ZIP with a SHA-256 manifest is stored in the established outputs directory.
- Config, tokenizer, index and installed modeling-source hashes are recorded.
  Weight-shard sizes and timestamps are recorded; **full weight-shard hashes
  are not calculated**. This is not checksum-bound model certification.
- No weights, vindex or persistent patch are changed. There is no new H1/H2,
  selectivity or paraphrase result.

## Next decision, not executed

Before interpreting H2 on NF4, check once whether the original asymmetric
symbol/atomic-number response to an entity-row contrast occurs on this
instrument. Port the fixed procedure and norm rule, not Q4K numeric vectors
or a tuned gain. If it fails, stop that transfer rather than searching gains.

If it succeeds, a controlled factorial test of residual intervention and PLE
replacement is more directly causal than changing donor elements, which also
changes native margins, tokenization and semantic distance. PLE-only,
residual-only, combined and complete-donor conditions would distinguish their
contributions. A PLE effect alone would not establish why the earlier residual
intervention was asymmetric. That next series needs its own frozen budget,
controls and stop rule.
